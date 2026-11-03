# ADR-003 — fronteira da orquestração

**Status:** aceito (O1/D3) · **Data:** 2026-11-02

## Contexto

Havia risco de a camada B-3PO virar um segundo runtime: dois donos de sessão, dois loops, transcript
duplicado. A ata exige um único dono por recurso.

## Decisão

A camada B-3PO é **dona do domínio**: registro versionado de bots, autorização, despacho, `execution_id` /
`correlation_id`, normalização de eventos e apresentação de estado.

O **Hermes é dono da execução**: loop, sessão, contexto, ferramentas, transcript, estado interno.

A camada não executa ferramenta, não cria loop, não duplica memória/transcript, não assume sessão e não
chama outro runtime. Aprovação e cancelamento passam por interface estreita ao Hermes.

## Enforcement

Política aplicada **no ponto de execução** (toolset desabilitado / aprovação existente). Se não houver
gancho suficiente, a capacidade fica desabilitada e emite `execution.failed` com motivo. Filtragem de UI
não conta como proteção.

## Consequências

- Um único dono por recurso (regra dura).
- Evento que o runtime não emite é declarado em `capabilities`, nunca fabricado.
- A UI deriva estado apenas de eventos confirmados.
