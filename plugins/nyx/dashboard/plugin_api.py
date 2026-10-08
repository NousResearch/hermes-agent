"""API do painel Nyx, montada em /api/plugins/nyx/.

``GET /eventos`` é um fluxo SSE: segue o barramento (``../barramento.py``) a partir do fim e manda
cada evento do Hermes assim que ele é escrito, de qualquer processo (CLI, gateway, TUI, dashboard).
``POST /ensaio`` escreve no MESMO barramento uma sequência de exemplo (pensar → ferramenta →
responder), pra ver o avatar reagir sem precisar de uma conversa de verdade.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse


def _carregar_barramento():
    # este arquivo é importado pelo caminho (não como pacote), então o irmão também vem pelo caminho
    nome = "hermes_nyx_barramento"
    if nome in sys.modules:
        return sys.modules[nome]
    spec = importlib.util.spec_from_file_location(nome, Path(__file__).resolve().parent.parent / "barramento.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nome] = mod
    spec.loader.exec_module(mod)
    return mod


barramento = _carregar_barramento()
router = APIRouter()

INTERVALO_S = 0.12
BATIMENTO_S = 15.0


async def _fluxo(request: Request) -> AsyncIterator[str]:
    deslocamento = barramento.fim_atual()
    ocioso = 0.0
    yield "retry: 1500\n\n"
    while not await request.is_disconnected():
        eventos, deslocamento = barramento.ler_desde(deslocamento)
        for ev in eventos:
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        if eventos:
            ocioso = 0.0
        else:
            ocioso += INTERVALO_S
            if ocioso >= BATIMENTO_S:
                ocioso = 0.0
                yield ": vivo\n\n"
        await asyncio.sleep(INTERVALO_S)


@router.get("/eventos")
async def eventos(request: Request) -> StreamingResponse:
    return StreamingResponse(
        _fluxo(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


_ENSAIO = (
    (0.0, "pensando", {}),
    (1.6, "ferramenta", {"ferramenta": "web_search", "id": "ensaio-1"}),
    (3.4, "ferramenta", {"ferramenta": "terminal", "id": "ensaio-2"}),
    (4.6, "ferramenta_fim", {"ferramenta": "web_search", "id": "ensaio-1", "erro": False}),
    (6.2, "ferramenta_fim", {"ferramenta": "terminal", "id": "ensaio-2", "erro": False}),
    (7.0, "falando", {}),
    (11.0, "pronto", {}),
)


async def _ensaiar() -> None:
    inicio = asyncio.get_running_loop().time()
    for quando, tipo, dados in _ENSAIO:
        espera = inicio + quando - asyncio.get_running_loop().time()
        if espera > 0:
            await asyncio.sleep(espera)
        barramento.publicar(tipo, sessao="ensaio", **dados)


@router.post("/ensaio")
async def ensaio() -> dict:
    asyncio.get_running_loop().create_task(_ensaiar())
    return {"ok": True, "duracao_s": _ENSAIO[-1][0]}
