"""Plugin nyx: os hooks do turno chegam, em ordem, a quem lê o barramento.

Carrega o plugin pela descoberta real (``PluginManager.discover_and_load``) num HERMES_HOME
temporário e confere o contrato entre os dois lados: o que os hooks publicam é exatamente o que
um leitor que conectou antes do turno recebe (o painel do dashboard é esse leitor).
"""

import importlib.util
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    (h / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": ["nyx"]}}))
    return h


def _barramento():
    spec = importlib.util.spec_from_file_location("nyx_barramento_teste", REPO / "plugins" / "nyx" / "barramento.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_turno_chega_ao_leitor_na_ordem_e_sem_conteudo(home):
    from hermes_cli import plugins as pmod

    mgr = pmod.PluginManager()
    mgr.discover_and_load()
    assert mgr._plugins["nyx"].enabled, mgr._plugins["nyx"].error

    bus = _barramento()
    inicio = bus.fim_atual()
    sid, turno = "s1", "t1"
    mgr.invoke_hook("pre_llm_call", session_id=sid, turn_id=turno, user_message="segredo do usuário")
    mgr.invoke_hook("pre_tool_call", tool_name="terminal", args={"command": "cat senha.txt"},
                    session_id=sid, tool_call_id="c1")
    mgr.invoke_hook("post_tool_call", tool_name="terminal", args={}, result="conteúdo sigiloso",
                    session_id=sid, tool_call_id="c1", status="error", duration_ms=12)
    for pedaco in ("Olá", ", tudo", " certo?"):
        mgr.invoke_hook("on_stream_delta", delta=pedaco, kind="text", session_id=sid, turn_id=turno, iteration=1)
    mgr.invoke_hook("post_llm_call", session_id=sid, turn_id=turno, assistant_response="Olá, tudo certo?")

    eventos, _ = bus.ler_desde(inicio)
    assert [e["tipo"] for e in eventos] == ["pensando", "ferramenta", "ferramenta_fim", "falando", "pronto"]
    fim = eventos[2]
    assert (fim["ferramenta"], fim["id"], fim["erro"]) == ("terminal", "c1", True)
    bruto = (home / "nyx" / "eventos.jsonl").read_text(encoding="utf-8")
    for vazamento in ("segredo", "senha", "sigiloso", "Olá"):
        assert vazamento not in bruto


def test_leitor_segue_o_barramento_depois_de_encurtado(home, monkeypatch):
    bus = _barramento()
    monkeypatch.setattr(bus, "LIMITE_BYTES", 2000)
    monkeypatch.setattr(bus, "CAUDA_LINHAS", 5)
    deslocamento = bus.fim_atual()
    vistos = []
    for i in range(60):
        bus.publicar("ferramenta", id=f"c{i}")
        novos, deslocamento = bus.ler_desde(deslocamento)
        vistos += [e["id"] for e in novos]
    # cada evento escrito é lido uma vez; quando o arquivo encolhe o leitor recomeça da cauda,
    # então pode rever eventos da cauda mas nunca perde o último
    assert vistos[-1] == "c59"
    assert set(vistos) >= {f"c{i}" for i in range(55, 60)}
