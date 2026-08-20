# PR infographic — Preserve Chromium sandbox in containers

> A imagem permanece incorporada na descrição do PR; o binário não é rastreado
> neste repositório.

## Arquitetura, em linguagem simples

Hermes controla ciclo de vida de sessões, mas usa `agent-browser` para executar
ações de navegação. No caminho antigo, ambos podiam participar do startup de
Chromium:

1. Hermes detectava Docker, root ou restrição AppArmor e adicionava
   `--no-sandbox`.
2. `agent-browser` 0.26.0 fazia nova detecção dentro do binário nativo e
   adicionava a mesma flag ao detectar `/.dockerenv`.
3. Mesmo um container não-root com user namespaces, AppArmor e seccomp
   funcionais perdia sandbox por causa da segunda camada.

O caminho novo, ativado explicitamente por
`AGENT_BROWSER_FORCE_SANDBOX=1`, dá startup de Chromium a Hermes:

```text
Hermes → Chromium com sandbox → CDP em 127.0.0.1 → agent-browser
```

`agent-browser` continua responsável pelas ações, snapshots e comandos. Como
ele recebe um endpoint CDP já existente, não executa seu launcher local de
Chromium e não consegue acrescentar `--no-sandbox` por detecção de Docker.

## Causa raiz corrigida

O problema não era apenas a condição em `_needs_chromium_sandbox_bypass()`.
Era uma decisão duplicada em processos diferentes: Hermes controlava uma
variável de ambiente, enquanto o binário nativo de `agent-browser` aplicava
sua própria política depois. Alterar apenas a primeira camada deixaria o bug
intacto.

## Garantias de segurança

- Modo explícito exige execução não-root.
- Endpoint CDP fica preso a `127.0.0.1` e usa porta efêmera.
- Perfil Chromium é único por sessão.
- Flags de bypass (`--no-sandbox`, `--no-zygote-sandbox` e equivalentes),
  além de flags que alterariam controles CDP/perfil, são rejeitadas.
- Falha de startup retorna diagnóstico acionável; não existe retry automático
  com sandbox desabilitado.
- Cleanup encerra Chromium Hermes-owned e remove perfil/socket da sessão.

## Validação

- `python -m py_compile tools/browser_tool.py` — passou em sandbox temporário.
- `python -m ruff check tools/browser_tool.py tests/tools/test_browser_sandbox.py`
  — passou.
- `pytest tests/tools/test_browser_sandbox.py` — 4 testes passaram em sandbox
  temporário isolado.
- Teste de integração real Docker/AppArmor não foi executado neste host
  Windows; deve ser executado em runner Linux não-root com user namespaces
  habilitados.

## Documentação atualizada

- `.env.example`
- `website/docs/reference/environment-variables.md`
- `website/docs/user-guide/features/browser.md`

## Geração da imagem

O arquivo foi gerado com ferramenta nativa de geração de imagem, usando tema
de diagrama técnico, caminho inseguro em vermelho e caminho sandbox-preserving
em verde.

O script prescrito `scripts/pr_infographic_prompt.py` não existe nesta branch;
o comando obrigatório foi tentado e falhou por arquivo ausente. Foi usado
prompt estruturado equivalente, sem adicionar helper ao código do PR.
