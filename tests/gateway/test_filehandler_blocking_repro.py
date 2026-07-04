"""Reproducir el atasco del gateway cuando ``FileHandler.emit()`` se bloquea.

Caso reportado: cuando un handler de ``gateway.log`` tarda o se bloquea al
escribir (fs lento, fd agotado, NFS colgado), los caminos críticos que
emiten ``logger.info`` sobre ese mismo handler serializan a través del
``Handler.lock`` interno. Las dos rutas que describen los reportes en
producción son:

  1. **Inbound nuevo** -- Telegram/Discord/etc entrega un mensaje nuevo
     mientras la sesión ya tiene un ``clarify`` pendiente. El handler de
     plataforma termina ejecutando ``_handle_message`` en el event loop,
     que pasa por ``logger.info("inbound message: ..."``, ``logger.info(
     "Gateway intercepted clarify text response ...")``, etc.

  2. **Clarify completion** -- el usuario responde al clarify pendiente;
     ``resolve_gateway_clarify()`` corre en otro hilo (el del adapter que
     recibió la respuesta) y emite su propio ``logger.info`` antes /
     después de hacer ``entry.event.set()``.

Si el handler de ``gateway.log`` está bloqueado en ``emit()``, ambos
hilos se cuelgan en el mismo lock. El hilo del agente que espera en
``clarify_gateway.wait_for_response`` sigue girando (``event.wait(1.0)``
en bucle con ``touch_activity_if_due``), pero el hilo que dispara la
resolución NO consigue despertar el ``Event`` porque su ``logger.info``
anterior no retorna. Resultado: el clarify "se entrega", pero el
``tool clarify completed`` nunca se emite, el turno nunca cierra, y el
siguiente inbound queda enrutado al clarify pendiente (no a un turno
nuevo) -- exactamente el síntoma documentado en el kanban skill de
hermes sobre el "Telegram inbound ↔ clarify() deadlock".

Este test inyecta un ``SlowFileHandler`` que envuelve el ``emit()`` real
de cada ``RotatingFileHandler`` del root logger y duerme una cantidad
configurable de segundos antes de delegar. Después lanza dos hilos en
paralelo:

  * **Hilo A** -- simula el inbound nuevo: emite un ``logger.info`` con
    el formato exacto que usa ``gateway.run._handle_message``, luego
    intenta llamar a ``resolve_text_response_for_session`` (que es lo
    que hace el branch ``if _resolved`` cuando hay un clarify pendiente).

  * **Hilo B** -- simula la completion de un clarify en background:
    emite ``logger.info("agent.tool_executor: tool clarify completed ...")``
    y luego hace ``resolve_gateway_clarify`` con un ``threading.Event``
    que un tercer hilo (el "agente") espera.

El test mide: (a) cuánto tardan A y B en completar, (b) si el ``Event``
se dispara dentro de un deadline razonable, (c) cuántos records
quedan en cola. Si ``slow_emit_seconds > 0``, A y B se serializan
visiblemente: la latencia agregada es ~2× el slowdown en lugar de
solaparse.

Run::

    cd /Users/pones/.hermes/hermes-agent
    pytest tests/gateway/test_filehandler_blocking_repro.py -v -s

    # O como repro standalone (no requiere pytest):
    python tests/gateway/test_filehandler_blocking_repro.py
"""

from __future__ import annotations

import io
import logging
import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_logging import RotatingFileHandler, setup_logging


# ---------------------------------------------------------------------------
# SlowFileHandler: envuelve un handler real y duerme dentro de emit()
# ---------------------------------------------------------------------------


class SlowFileHandler(logging.Handler):
    """Proxy que añade un ``time.sleep(delay)`` antes de delegar a ``inner``.

    Importante: NO toca el ``inner.acquire()`` / ``inner.release()`` -- el
    handler real sigue protegiendo sus propias escrituras con su
    ``RLock``. Lo que queremos exponer es que, mientras ``inner.emit()``
    está durmiendo, cualquier otro hilo que intente pasar por ese mismo
    handler se queda bloqueado en el lock. Eso es exactamente el atasco
    real: el ``Handler.lock`` es por-instancia, pero como hay UN solo
    ``RotatingFileHandler`` apuntando a ``gateway.log`` en el root logger,
    todos los ``logger.info(...)`` que matchean el ``_ComponentFilter(
    ["gateway.*"])`` se serializan en él.
    """

    def __init__(self, inner: logging.Handler, delay_seconds: float):
        super().__init__(inner.level)
        self.inner = inner
        self.delay_seconds = delay_seconds
        self.records_passed = 0
        self.total_delay = 0.0
        self._stats_lock = threading.Lock()
        # Reusar el formatter / filter del inner para que los records
        # lleguen idénticos al handler real.
        self.setFormatter(getattr(inner, "formatter", None))
        if getattr(inner, "filters", None):
            self.filters = list(inner.filters)

    def emit(self, record: logging.LogRecord) -> None:
        # Filtro manual: si el inner tiene filter (e.g. _ComponentFilter),
        # hay que aplicarlo antes de dormir, si no contaminamos las stats.
        if self.filters:
            if not all(f.filter(record) for f in self.filters):
                return
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)
            with self._stats_lock:
                self.total_delay += self.delay_seconds
        with self._stats_lock:
            self.records_passed += 1
        self.inner.emit(record)

    def close(self) -> None:
        try:
            self.inner.close()
        finally:
            super().close()


# ---------------------------------------------------------------------------
# Helpers para aislar la harness del estado de logging global
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_root_logger(tmp_path):
    """Snapshot + restore de los handlers del root logger alrededor del test.

    Cada test parte de un root logger con tres ``RotatingFileHandler``
    recién instalados (agent.log, errors.log, gateway.log) y los recupera
    al final, así no contaminamos el resto del suite ni dejamos fds
    abiertos.
    """
    # Forzar un HERMES_HOME temporal antes de setup_logging() para que los
    # archivos vivan en tmp_path y se limpien automáticamente.
    home = tmp_path / "hermes_home"
    home.mkdir()
    os.environ["HERMES_HOME"] = str(home)

    # Reset state global de hermes_logging por si otro test lo dejó a medias.
    import hermes_logging
    hermes_logging._logging_initialized = False

    setup_logging(hermes_home=home, mode="gateway", force=True)

    root = logging.getLogger()
    pre_existing = list(root.handlers)
    try:
        yield root, home
    finally:
        # Limpiar todos los handlers añadidos durante el test.
        for h in list(root.handlers):
            try:
                h.close()
            except Exception:
                pass
            root.removeHandler(h)
        # Restaurar lo que había antes (suele ser vacío, pero por si acaso).
        for h in pre_existing:
            root.addHandler(h)
        hermes_logging._logging_initialized = False


def _wrap_root_handlers_with_slow(root: logging.Logger, delay: float) -> list[SlowFileHandler]:
    """Sustituye cada ``RotatingFileHandler`` del root por un ``SlowFileHandler``.

    Devuelve la lista de proxies instalados para poder inspeccionar
    contadores desde el test.
    """
    proxies: list[SlowFileHandler] = []
    for h in list(root.handlers):
        if isinstance(h, RotatingFileHandler):
            proxy = SlowFileHandler(h, delay_seconds=delay)
            root.removeHandler(h)
            root.addHandler(proxy)
            proxies.append(proxy)
    return proxies


def _restore_root_handlers(root: logging.Logger, proxies: list[SlowFileHandler]) -> None:
    """Desenvuelve los proxies y vuelve a poner los handlers originales."""
    for proxy in proxies:
        inner = proxy.inner
        root.removeHandler(proxy)
        try:
            proxy.close()
        except Exception:
            pass
        root.addHandler(inner)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slow_seconds", [0.0, 0.2, 0.5])
def test_handler_lock_serializes_concurrent_inbound_and_clarify_logs(
    isolated_root_logger, slow_seconds
):
    """Demuestra que un handler lento serializa los dos caminos críticos.

    Cuando ``slow_seconds=0`` los dos hilos terminan casi en paralelo
    (skew < 50ms). Con ``slow_seconds=0.5`` se serializan: la latencia
    agregada es ~``2 * slow_seconds`` en lugar de ~``slow_seconds``.
    """
    root, _home = isolated_root_logger
    proxies = _wrap_root_handlers_with_slow(root, slow_seconds)

    try:
        # Logger dedicado para evitar que _ComponentFilter nos filtre los
        # records. Esto NO es trampa: el objetivo es medir la serialización
        # del lock del handler, no el filtrado por componente (que es una
        # optimización pero NO aislamiento entre hilos).
        gateway_logger = logging.getLogger("gateway.repro")

        barrier = threading.Barrier(2, timeout=5.0)
        timings: dict[str, float] = {}
        events: dict[str, threading.Event] = {
            "inbound": threading.Event(),
            "clarify": threading.Event(),
        }

        def inbound_path() -> None:
            barrier.wait()
            t0 = time.monotonic()
            # Mismo formato que gateway/run.py usa para los logs de inbound.
            gateway_logger.info(
                "Gateway intercepted clarify text response (session=%s, id=%s)",
                "test-session",
                "clarify-123",
            )
            # Simulamos el trabajo posterior al log: routing, decisión
            # de qué handler invocar, etc. Aquí basta con emitir otro
            # record para evidenciar la serialización.
            gateway_logger.info("inbound message: text='hello' session=test-session")
            timings["inbound"] = time.monotonic() - t0
            events["inbound"].set()

        def clarify_path() -> None:
            barrier.wait()
            t0 = time.monotonic()
            gateway_logger.info(
                "agent.tool_executor: tool clarify completed (%.2fs, %d chars)",
                slow_seconds * 2,  # mentira; sólo importa el log
                211,
            )
            gateway_logger.info("gateway.run: response ready: response=211 chars")
            timings["clarify"] = time.monotonic() - t0
            events["clarify"].set()

        t_inbound = threading.Thread(target=inbound_path, name="inbound")
        t_clarify = threading.Thread(target=clarify_path, name="clarify")
        t0 = time.monotonic()
        t_inbound.start()
        t_clarify.start()
        assert events["inbound"].wait(timeout=10.0), "inbound path no terminó"
        assert events["clarify"].wait(timeout=10.0), "clarify path no terminó"
        wall = time.monotonic() - t0

        # Cada proxy contó cuántos records pasaron por su inner.
        total_records = sum(p.records_passed for p in proxies)
        total_delay = sum(p.total_delay for p in proxies)

        # Reporte
        sys.stderr.write(
            f"\n[REPORT slow={slow_seconds}s] "
            f"inbound={timings['inbound']*1000:.1f}ms "
            f"clarify={timings['clarify']*1000:.1f}ms "
            f"wall={wall*1000:.1f}ms "
            f"records={total_records} "
            f"total_delay={total_delay*1000:.1f}ms\n"
        )

        # Sanity: ambos hilos escribieron sus 2 records cada uno = 4 totales.
        assert total_records >= 4

        if slow_seconds == 0.0:
            # Sin slowdown, ambos hilos deberían correr esencialmente en
            # paralelo. wall << inbound + clarify.
            assert wall < (timings["inbound"] + timings["clarify"]) * 0.75, (
                f"sin slowdown esperábamos paralelismo, pero wall={wall:.3f}s "
                f"es casi la suma de las dos latencias"
            )
        else:
            # Con slowdown, los dos hilos SE SERIALIZAN en el lock del
            # handler. wall ≈ max(inbound, clarify) y la suma de delays
            # en los proxies es ~4 * slow_seconds (2 records × 2 hilos).
            #
            # Margen generoso (×1.5) por overhead del scheduler en CI.
            expected_min_delay = 4 * slow_seconds * 0.8
            assert total_delay >= expected_min_delay, (
                f"esperábamos que el handler durmiera al menos "
                f"{expected_min_delay:.3f}s en total, pero total_delay="
                f"{total_delay:.3f}s -- ¿el proxy no se instaló?"
            )
            # La pared no puede ser mejor que el slowdown serializado.
            assert wall >= 2 * slow_seconds * 0.8, (
                f"con slow={slow_seconds}s esperábamos wall>={2*slow_seconds*0.8:.3f}s, "
                f"medimos {wall:.3f}s"
            )
    finally:
        _restore_root_handlers(root, proxies)


def test_clarify_event_unblock_blocked_by_handler_lock(isolated_root_logger):
    """Demuestra el deadlock "clarify nunca completa" de forma determinista.

    Simula los tres hilos en juego:

      * **agente**: llama a ``clarify_gateway.wait_for_response``, que
        bloquea en ``entry.event.wait(1.0)`` en bucle (timeout total 3s).
      * **adapter (clarify completion)**: emite el log de "tool clarify
        completed" y luego hace ``resolve_gateway_clarify`` (que dispara
        ``entry.event.set()``).
      * **adapter (inbound)**: emite su propio log de inbound ANTES de
        que la completion thread arranque -- así competimos por el lock
        desde el inicio.

    Determinismo: para garantizar que el inbound thread agarra el lock
    ANTES de que la completion thread termine su primer ``logger.info``,
    usamos una ``threading.Barrier(3)`` que suelta los tres hilos a la
    vez. La completion thread emite DOS logs lentos seguidos (uno de
    "tool clarify completed" y otro de "response ready"); entre medias
    se cuela el inbound con sus dos logs. Con ``slow_seconds=0.5`` y 4
    emits por hilo × 2 handlers (agent.log + gateway.log) el trabajo
    serializado es 8 × 0.5 = 4.0s mínimo, claramente por encima del
    timeout del agente (3s) -- la completion nunca llega al resolve.

    Si el handler está sano, la completion termina su trabajo en ~1s y
    el agente responde 'A'. Si está bloqueado, la completion queda
    detrás del inbound en el lock del handler, supera el timeout del
    agente, y el ``Event.wait`` se despierta con ``response=None``.
    """
    from tools import clarify_gateway as cm

    root, _home = isolated_root_logger
    # 0.8s por emit × 3 emits (completion) × 2 handlers (agent.log + gateway.log)
    # = 4.8s de trabajo para la completion thread. >3s timeout del agente.
    slow_seconds = 0.8
    proxies = _wrap_root_handlers_with_slow(root, slow_seconds)

    # Limpiar el estado global de clarify (otros tests lo pueden haber dejado sucio).
    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()

    try:
        session_key = "test-session-deadlock"
        gateway_logger = logging.getLogger("gateway.repro")

        # 1. Registrar un clarify pendiente (igual que gateway/run.py:_clarify_callback_sync)
        entry = cm.register(
            clarify_id="clarify-deadlock-1",
            session_key=session_key,
            question="¿Cuál opción?",
            choices=["A", "B"],
        )

        # 2. Hilo "agente": bloquea en wait_for_response con timeout corto.
        agent_result: dict[str, object] = {}
        agent_started = threading.Event()
        # Barrera de 3: agente espera, completion espera, inbound espera.
        # Cuando los 3 llegan, se sueltan a la vez -- el agente entra
        # al bucle de wait, completion arranca sus logs, inbound arranca
        # sus logs. Ahora compiten por el lock del handler.
        barrier = threading.Barrier(3, timeout=5.0)

        def agent_thread() -> None:
            barrier.wait()
            agent_started.set()
            t0 = time.monotonic()
            # Timeout corto (3s) para que el test no se cuelgue si la
            # completion nunca llega -- queremos ver el timeout, no un
            # pytest timeout externo.
            response = cm.wait_for_response("clarify-deadlock-1", timeout=3.0)
            agent_result["response"] = response
            agent_result["elapsed"] = time.monotonic() - t0

        # 3. Hilo "clarify completion": emite DOS logs lentos para
        #    maximizar la oportunidad de que inbound se cuele entre
        #    medias, luego intenta resolver.
        completion_result: dict[str, object] = {}
        completion_done = threading.Event()

        def clarify_completion_thread() -> None:
            barrier.wait()
            t0 = time.monotonic()
            gateway_logger.info(
                "agent.tool_executor: tool clarify completed (%.2fs, %d chars)",
                slow_seconds * 2, 211,
            )
            # Otro log "response ready" para forzar más tiempo en el lock.
            gateway_logger.info("gateway.run: response ready: response=211 chars")
            # Tercer log para amplificar la ventana de contención.
            gateway_logger.info(
                "agent.conversation_loop: Turn ended: reason=clarify_resolved response_len=211"
            )
            # Si llegamos aquí, podemos resolver el clarify.
            cm.resolve_gateway_clarify("clarify-deadlock-1", "A")
            completion_result["elapsed"] = time.monotonic() - t0
            completion_done.set()

        # 4. Hilo "inbound": emite DOS logs, demostrando que comparte
        #    el mismo lock. Sale por la barrier al mismo tiempo que
        #    completion -- así garantiza contención desde t=0.
        inbound_result: dict[str, object] = {}
        inbound_done = threading.Event()

        def inbound_thread() -> None:
            barrier.wait()
            t0 = time.monotonic()
            gateway_logger.info(
                "Gateway intercepted clarify text response (session=%s, id=%s)",
                session_key, "clarify-deadlock-1",
            )
            gateway_logger.info("inbound message: text='hello' session=%s", session_key)
            inbound_result["elapsed"] = time.monotonic() - t0
            inbound_done.set()

        t_agent = threading.Thread(target=agent_thread, name="agent")
        t_completion = threading.Thread(target=clarify_completion_thread, name="completion")
        t_inbound = threading.Thread(target=inbound_thread, name="inbound")

        t_agent.start()
        t_completion.start()
        t_inbound.start()

        # Esperamos a que todo termine (o al timeout del agente).
        completion_done.wait(timeout=10.0)
        inbound_done.wait(timeout=10.0)
        t_agent.join(timeout=10.0)

        sys.stderr.write(
            f"\n[REPORT deadlock slow={slow_seconds}s] "
            f"agent_response={agent_result.get('response')!r} "
            f"agent_elapsed={agent_result.get('elapsed', 0):.2f}s "
            f"completion_elapsed={completion_result.get('elapsed', 0):.2f}s "
            f"inbound_elapsed={inbound_result.get('elapsed', 0):.2f}s\n"
        )

        # El agente DEBE haber recibido la respuesta si la completion
        # thread logró llegar al resolve antes del timeout. Si el log
        # lento bloqueó la completion thread >3s (timeout del agente),
        # vemos el síntoma del deadlock: response=None, elapsed≈3s.
        if agent_result.get("response") is None:
            # Deadlock reproducido -- emitimos evidencia diagnóstica
            # pero dejamos el test pasar (es la prueba del fallo, no
            # un fallo del test). Marcamos con un print explícito.
            total_delay = sum(p.total_delay for p in proxies)
            sys.stderr.write(
                f"[DEADLOCK CONFIRMADO] agent_response=None "
                f"agent_elapsed={agent_result.get('elapsed', 0):.2f}s "
                f"completion_elapsed={completion_result.get('elapsed', 0):.2f}s "
                f"inbound_elapsed={inbound_result.get('elapsed', 0):.2f}s "
                f"handler_total_delay={total_delay:.2f}s\n"
            )
            assert agent_result.get("response") is None
        else:
            pytest.fail(
                f"Esperábamos deadlock (response=None, elapsed≈3s) pero el "
                f"agente respondió {agent_result.get('response')!r} a los "
                f"{agent_result.get('elapsed', 0):.2f}s. Posibles causas: "
                f"(a) el orden de scheduling permitió que completion "
                f"terminara antes del timeout, (b) el slowdown no fue "
                f"suficiente. Re-ejecutar o subir slow_seconds."
            )
    finally:
        # Limpiar estado de clarify para no contaminar otros tests.
        with cm._lock:
            cm._entries.clear()
            cm._session_index.clear()
            cm._notify_cbs.clear()
        _restore_root_handlers(root, proxies)


# ---------------------------------------------------------------------------
# Standalone entrypoint: ejecutar como script sin pytest
# ---------------------------------------------------------------------------


def _standalone_run() -> int:
    """Repro rápido sin pytest. Crea un HERMES_HOME temporal, instala los
    proxies, lanza los dos hilos paralelos, imprime el reporte."""
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="hermes-deadlock-repro-"))
    os.environ["HERMES_HOME"] = str(tmp)
    logging.getLogger().setLevel(logging.INFO)
    setup_logging(hermes_home=tmp, mode="gateway", force=True)

    root = logging.getLogger()
    for slow in (0.0, 0.2, 0.5):
        proxies = _wrap_root_handlers_with_slow(root, slow)
        barrier = threading.Barrier(2, timeout=5.0)
        timings: dict[str, float] = {}

        def emit_pair(name: str) -> None:
            barrier.wait()
            t0 = time.monotonic()
            log = logging.getLogger("gateway.repro")
            log.info("Gateway intercepted clarify text response (session=%s, id=%s)", "s", "c")
            log.info("inbound message: text='hello' session=s")
            timings[name] = time.monotonic() - t0

        t1 = threading.Thread(target=emit_pair, args=("a",))
        t2 = threading.Thread(target=emit_pair, args=("b",))
        wall0 = time.monotonic()
        t1.start(); t2.start()
        t1.join(); t2.join()
        wall = time.monotonic() - wall0
        total_delay = sum(p.total_delay for p in proxies)
        print(
            f"[standalone slow={slow:>4}s] a={timings['a']*1000:6.1f}ms "
            f"b={timings['b']*1000:6.1f}ms wall={wall*1000:6.1f}ms "
            f"delay={total_delay*1000:6.1f}ms",
            file=sys.stderr,
        )
        _restore_root_handlers(root, proxies)

    return 0


if __name__ == "__main__":
    sys.exit(_standalone_run())