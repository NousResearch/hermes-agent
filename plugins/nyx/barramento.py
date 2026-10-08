"""Barramento de eventos do Nyx: um JSONL curto em ``$HERMES_HOME/nyx/eventos.jsonl``.

Os hooks rodam no processo que roda o agente (CLI, gateway, TUI, dashboard); o painel roda no
processo do dashboard. Um arquivo só-acréscimo é o meio comum mais simples entre eles: cada evento
é uma linha (bem menor que PIPE_BUF, então a escrita com O_APPEND é atômica entre processos) e o
leitor guarda o deslocamento. Quando passa do limite o arquivo é reescrito só com a cauda; o leitor
percebe pelo tamanho menor que o seu deslocamento e recomeça do início.

O home é resolvido a cada chamada (``get_hermes_home``), nunca guardado: um processo pode servir
vários perfis e cada evento vai pro home do perfil que está em escopo naquele turno.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from hermes_constants import get_hermes_home

LIMITE_BYTES = 256 * 1024
CAUDA_LINHAS = 200
_trava = threading.Lock()


def caminho() -> Path:
    return get_hermes_home() / "nyx" / "eventos.jsonl"


def publicar(tipo: str, **dados: Any) -> None:
    """Acrescenta um evento. Nunca levanta: o avatar é decoração e não pode atrapalhar o turno."""
    linha = json.dumps({"t": round(time.time(), 3), "tipo": tipo, **dados}, ensure_ascii=False) + "\n"
    arq = caminho()
    try:
        with _trava:
            arq.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(arq, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, linha.encode("utf-8"))
                tamanho = os.fstat(fd).st_size
            finally:
                os.close(fd)
            if tamanho > LIMITE_BYTES:
                _encurtar(arq)
    except OSError:
        pass


def _encurtar(arq: Path) -> None:
    linhas = arq.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-CAUDA_LINHAS:]
    tmp = arq.with_suffix(".tmp")
    tmp.write_text("".join(linhas), encoding="utf-8")
    os.replace(tmp, arq)


def ler_desde(deslocamento: int) -> Tuple[List[Dict[str, Any]], int]:
    """Eventos novos a partir de *deslocamento* (bytes) e o novo deslocamento.

    Linha incompleta no fim (escrita em andamento) fica pra próxima leitura. Arquivo encurtado
    (tamanho < deslocamento) recomeça do zero.
    """
    arq = caminho()
    try:
        tamanho = arq.stat().st_size
    except OSError:
        return [], 0
    if tamanho < deslocamento:
        deslocamento = 0
    if tamanho == deslocamento:
        return [], deslocamento
    with arq.open("rb") as f:
        f.seek(deslocamento)
        bruto = f.read(tamanho - deslocamento)
    fim = bruto.rfind(b"\n") + 1
    eventos: List[Dict[str, Any]] = []
    for linha in bruto[:fim].splitlines():
        try:
            ev = json.loads(linha)
        except ValueError:
            continue
        if isinstance(ev, dict):
            eventos.append(ev)
    return eventos, deslocamento + fim


def fim_atual() -> int:
    """Deslocamento do fim do arquivo: quem acabou de conectar só quer o que vier daqui pra frente."""
    try:
        return caminho().stat().st_size
    except OSError:
        return 0
