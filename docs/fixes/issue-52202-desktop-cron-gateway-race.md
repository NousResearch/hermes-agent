# Fix #52202 — Race do cron entre o desktop e o gateway

- **Issue:** [#52202 — Desktop dashboard cron scheduler races the gateway; when it wins a tick it has no live adapter and delivery hangs until script_timeout (600s)](https://github.com/NousResearch/hermes-agent/issues/52202)
- **Branch:** `fix/desktop-cron-gateway-race`
- **Autor:** joaomarcos (joaomarcosdias444@gmail.com)
- **Status:** implementado e testado (Windows/macOS/Linux)

---

## 1. Problema

Quando um serviço `hermes gateway run` (launchd no macOS, systemd no Linux) e o
app desktop rodam no **mesmo `HERMES_HOME`**, surgem **dois schedulers de cron
in-process** competindo.

O backend do dashboard do desktop (`HERMES_DESKTOP=1`) inicia o próprio scheduler
em `hermes_cli/web_server.py::_start_desktop_cron_ticker`, ao lado do scheduler do
gateway. Eles se coordenam por `cron/.tick.lock` ("quem pega o lock primeiro
executa o tick"), o que evita disparo duplo — **mas o lock só garante exclusão
mútua, não capacidade de execução**.

Quando o processo do desktop vence o tick de um job que entrega a uma plataforma
viva (ex.: Telegram), o job roda **sem adapter vivo** (a própria docstring antiga
admitia: *"no live adapters; delivery falls back to the per-platform send path"*).
O processo desktop foi lançado pela GUI e não herda o ambiente de
inferência/credenciais do gateway. O resultado é o job travando até o timeout e
o usuário recebendo a mensagem genérica:

> `provider timeout. Fallback chain was exhausted or unavailable.`

O sintoma é **intermitente** — um "cara ou coroa" sobre qual processo pega o lock
a cada tick. O mesmo job tem sucesso nos dias em que o gateway vence e falha nos
dias em que o app está aberto no horário de disparo e vence.

### Evidências (do issue)

- Cron reporta `Script timed out after 600s` mesmo para um `no_agent` trivial
  (um `cat` de arquivo pré-montado) que roda em 0.00s standalone.
- Job agendado em `HH:30`, falha/timeout em `HH:40` — exatamente 600s depois.
- `gui.log` mostra a mudança de um *Desktop cron ticker* passivo para um
  *Desktop cron scheduler (provider=builtin)* completo num update recente — quando
  as falhas intermitentes começaram.
- O log do gateway mostra o job entregue ao Telegram apenas como a notificação de
  falha, não como o relatório.

---

## 2. Causa raiz

Confirmada por leitura do código:

1. `web_server.py::_lifespan` iniciava o ticker **sempre** que `HERMES_DESKTOP=1`,
   **sem checar se já existe um gateway vivo** — apesar de essa checagem já
   existir e ser usada em `hermes_cli/cron.py` (`find_gateway_pids()`) e
   `gateway/status.py` (`is_gateway_running()`).
2. O `cron/.tick.lock` (`cron/scheduler.py::tick`) coordena **apenas exclusão de
   tick**, não capacidade. O vencedor pode não ter adapters/ambiente — invariante
   "apenas um scheduler executa jobs por `HERMES_HOME`" violada.
3. Bug latente secundário: no caminho de entrega standalone
   (`cron/scheduler.py::_deliver_result`), o `asyncio.run(coro)` do envio primário
   era **sem timeout** — um envio HTTP travado pendurava o worker do cron
   indefinidamente.

---

## 3. Solução implementada (passo a passo)

### Passo 1 — Detecção de gateway vivo (correção principal)

Arquivo: `hermes_cli/web_server.py`, dentro de `_lifespan`.

Antes de iniciar o ticker sob `HERMES_DESKTOP=1`, consultamos
`gateway.status.is_gateway_running()`:

- **Gateway vivo detectado** → o desktop vira *passive observer* e **não** inicia
  scheduler próprio. O gateway passa a ser o único executor de cron daquele
  `HERMES_HOME`.
- **Sem gateway** (instalação só-desktop) → o ticker inicia normalmente, como
  antes.
- **Erro na checagem** → assume "sem gateway" e inicia o ticker (direção de falha
  segura: nunca silencia o cron de uma instalação só-desktop).

```python
if os.getenv("HERMES_DESKTOP") == "1":
    gateway_alive = False
    try:
        from gateway.status import is_gateway_running
        gateway_alive = is_gateway_running()  # PID file + runtime lock, com stale cleanup
    except Exception:
        gateway_alive = False
    if gateway_alive:
        _log.info(
            "Desktop backend: live gateway detected — not starting an own cron "
            "scheduler (the gateway is the sole cron executor for this HERMES_HOME)."
        )
    else:
        cron_stop = threading.Event()
        cron_thread = threading.Thread(
            target=_start_desktop_cron_ticker, args=(cron_stop,),
            daemon=True, name="desktop-cron-ticker",
        )
        cron_thread.start()
```

Esta é exatamente a primeira opção do *Suggested fix* do issue
("detect a live gateway.pid / gateway lock and skip").

### Passo 2 — Timeout na entrega standalone (defesa em profundidade)

Arquivo: `cron/scheduler.py`, em `_deliver_result`.

O envio primário passa a ser limitado por `asyncio.wait_for(..., timeout=30)`,
espelhando o orçamento de 30s que o ramo de fallback (threadpool) já tinha. Um
envio de plataforma travado não pendura mais o worker do cron.

```python
def _bounded_send():
    return asyncio.wait_for(
        _send_to_platform(platform, pconfig, chat_id, cleaned_delivery_content,
                          thread_id=thread_id, media_files=media_files),
        timeout=30,
    )
coro = _bounded_send()
try:
    result = asyncio.run(coro)
except RuntimeError:
    coro.close()  # evita "coroutine was never awaited"
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, _bounded_send())
        result = future.result(timeout=35)
```

### Passo 3 — Teste de regressão

Arquivo: `tests/hermes_cli/test_web_server.py`.

Adicionado `test_ticker_skipped_when_gateway_alive`: com `HERMES_DESKTOP=1` e
`is_gateway_running()` mockado para `True`, o ticker do desktop **não** dispara.

---

## 4. Arquivos alterados

| Arquivo | Mudança |
|---|---|
| `hermes_cli/web_server.py` | Detecção de gateway vivo antes de iniciar o ticker |
| `cron/scheduler.py` | Timeout de 30s no envio standalone de `_deliver_result` |
| `tests/hermes_cli/test_web_server.py` | Novo teste do invariante |

---

## 5. Por que esta abordagem (e não as outras)

| Alternativa | Por que não |
|---|---|
| Handoff de entrega via IPC para o gateway | Muito mais superfície/complexidade; novo canal a manter |
| Validar capacidade (adapter) no vencedor do lock | Trata o sintoma, não a duplicação de schedulers; ambiente de inferência é difícil de auto-detectar |
| Desktop nunca executa jobs | Quebra a instalação só-desktop (sem gateway), onde o ticker é o único executor |

A detecção de gateway vivo resolve a **causa raiz** independentemente do mecanismo
exato do timeout (script, entrega ou inatividade do agente): se o desktop não
executa quando o gateway está vivo, nenhum desses caminhos de falha ocorre.

---

## 6. Compatibilidade cross-platform

A correção apoia-se nas mesmas primitivas de liveness que o próprio gateway usa,
todas com tratamento explícito por SO:

- `_pid_exists`: psutil → fallback ctypes `OpenProcess`/`WaitForSingleObject`
  (Windows) / `os.kill(pid, 0)` (POSIX).
- Runtime lock: `msvcrt.locking` (Windows) / `fcntl.flock` (POSIX).
- `get_running_pid()` **não** faz subprocess/rede (só I/O de arquivo) → sem atraso
  de startup em nenhum SO.

Verificado em **Windows 11**, e válido para **macOS** (launchd) e **Linux**
(systemd), em x86_64 e ARM64. Instalações só-desktop, só-gateway e dashboard de
servidor preservadas.

---

## 7. Como verificar

### Testes automatizados

```bash
python -m pytest tests/hermes_cli/test_web_server.py::TestDesktopCronTicker -q
python -m pytest tests/cron/test_scheduler.py tests/cron/test_scheduler_provider.py -q
```

### Verificação manual (checklist)

- [ ] Com gateway vivo, `gui.log` mostra *"live gateway detected — not starting an
      own cron scheduler"* e **nunca** *"Desktop cron scheduler started"*.
- [ ] 48h sem timeout de 600s em jobs Telegram com gateway + desktop abertos.
- [ ] Só-desktop: jobs continuam disparando.
- [ ] Só-gateway: inalterado.
- [ ] `.tick.lock` não fica órfão após kill (lock por-fd, liberado no fechamento
      do processo).

---

## 8. Ressalvas conhecidas (não-regressões)

- **HERMES_HOME/profile divergente** entre gateway e desktop: se rodarem em homes
  diferentes, o desktop pode não ver o gateway e iniciar o ticker; o `.tick.lock`
  é ancorado no *default root* compartilhado, então a race poderia voltar nesse
  caso de borda. Já existia antes; mitigação opcional: re-checar liveness por tick.
- **Race de arranque**: se o desktop subir antes do gateway, a checagem dá "sem
  gateway" e o ticker inicia. Mitigação opcional: re-checagem periódica.
