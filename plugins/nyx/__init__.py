"""Nyx — avatar de partículas do Hermes.

Os hooks do turno viram eventos curtos no barramento (``barramento.py``); o painel Nyx do
dashboard (``dashboard/``) lê esses eventos e desenha o estado: em repouso só o busto; a galáxia
nasce quando o Hermes começa a pensar, cada ferramenta solta um cometa, e tudo recolhe quando a
resposta começa a sair.

Só observa: nenhum hook devolve diretiva, nenhuma ferramenta nova, nada do conteúdo da conversa
(argumentos, resultados, texto) sai do turno — só nomes de ferramenta, ids e status.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

from . import barramento

# primeira fala de cada (turno, iteração): os deltas de texto chegam aos montes, o avatar só precisa
# saber quando a resposta COMEÇOU a sair
_falas_vistas: "OrderedDict[tuple, None]" = OrderedDict()
_trava = threading.Lock()


def _ao_pensar(session_id: str = "", turn_id: str = "", **_: Any) -> None:
    barramento.publicar("pensando", sessao=session_id, turno=turn_id)


def _ao_chamar(tool_name: str = "", tool_call_id: str = "", session_id: str = "", **_: Any) -> None:
    barramento.publicar("ferramenta", sessao=session_id, ferramenta=tool_name, id=tool_call_id)


def _ao_terminar(tool_name: str = "", tool_call_id: str = "", session_id: str = "", status: str = "",
                 duration_ms: Any = None, **_: Any) -> None:
    barramento.publicar("ferramenta_fim", sessao=session_id, ferramenta=tool_name, id=tool_call_id,
                        erro=status == "error", ms=duration_ms)


def _ao_escrever(kind: str = "", session_id: str = "", turn_id: str = "", iteration: int = 0, **_: Any) -> None:
    if kind != "text":
        return
    chave = (session_id, turn_id, iteration)
    with _trava:
        if chave in _falas_vistas:
            return
        _falas_vistas[chave] = None
        while len(_falas_vistas) > 256:
            _falas_vistas.popitem(last=False)
    barramento.publicar("falando", sessao=session_id, turno=turn_id)


def _ao_responder(session_id: str = "", turn_id: str = "", **_: Any) -> None:
    barramento.publicar("pronto", sessao=session_id, turno=turn_id)


def _ao_delegar(parent_session_id: str = "", child_session_id: str = "", **_: Any) -> None:
    barramento.publicar("subagente", sessao=parent_session_id or "", filho=child_session_id or "")


def _ao_voltar(parent_session_id: str = "", child_session_id: str = "", child_status: str = "", **_: Any) -> None:
    barramento.publicar("subagente_fim", sessao=parent_session_id or "", filho=child_session_id or "",
                        status=child_status or "")


_HOOKS = {
    "pre_llm_call": _ao_pensar,
    "pre_tool_call": _ao_chamar,
    "post_tool_call": _ao_terminar,
    "on_stream_delta": _ao_escrever,
    "post_llm_call": _ao_responder,
    "subagent_start": _ao_delegar,
    "subagent_stop": _ao_voltar,
}


def register(ctx) -> None:
    for nome, fn in _HOOKS.items():
        ctx.register_hook(nome, fn)
