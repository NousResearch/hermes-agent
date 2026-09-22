<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤
<p align="center">
  <a href="https://hermes-agent.nousresearch.com/">Hermes Agent</a> | <a href="https://hermes-agent.nousresearch.com/">Hermes Desktop</a>
</p>
<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Documentação"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/Licen%C3%A7a-MIT-green?style=for-the-badge" alt="Licença: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Criado%20por-Nous%20Research-blueviolet?style=for-the-badge" alt="Criado por Nous Research"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/Lang-English-blue?style=for-the-badge" alt="English"></a>
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/Lang-中文-red?style=for-the-badge" alt="中文"></a>
  <a href="README.ur-pk.md"><img src="https://img.shields.io/badge/Lang-اردو-green?style=for-the-badge" alt="اردو"></a>
  <a href="README.es.md"><img src="https://img.shields.io/badge/Lang-Espa%C3%B1ol-orange?style=for-the-badge" alt="Español"></a>
</p>

**O agente de IA com aprimoramento contínuo criado pela [Nous Research](https://nousresearch.com).** É o único agente com um ciclo de aprendizado integrado — ele cria skills (habilidades) a partir da experiência, as aprimora durante o uso, se incentiva a persistir conhecimento, pesquisa as próprias conversas passadas e constrói um modelo cada vez mais profundo de quem você é ao longo das sessões. Rode em um VPS de US$ 5, em um cluster de GPUs ou em infraestrutura serverless que custa quase nada quando ociosa. Ele não fica preso ao seu laptop — converse com ele pelo Telegram enquanto ele trabalha em uma VM na nuvem.

Use qualquer modelo que você quiser — [Nous Portal](https://portal.nousresearch.com), OpenRouter, OpenAI, o seu próprio endpoint e [muitos outros](https://hermes-agent.nousresearch.com/docs/integrations/providers). Troque com `hermes model` — sem mudanças de código, sem lock-in.

<table>
<tr><td><b>Uma interface de terminal de verdade</b></td><td>TUI completa com edição multilinha, autocompletar de comandos de barra, histórico de conversas, interrupção e redirecionamento, e saída de ferramentas em streaming.</td></tr>
<tr><td><b>Vive onde você vive</b></td><td>Telegram, Discord, Slack, WhatsApp, Signal e CLI — tudo a partir de um único processo gateway. Transcrição de notas de voz, continuidade de conversas entre plataformas.</td></tr>
<tr><td><b>Um ciclo de aprendizado fechado</b></td><td>Memória curada pelo agente com lembretes periódicos. Criação autônoma de skills após tarefas complexas. As skills se aprimoram durante o uso. Busca de sessões com FTS5 e sumarização por LLM para recuperação entre sessões. Modelagem dialética de usuários com <a href="https://github.com/plastic-labs/honcho">Honcho</a>. Compatível com o padrão aberto do <a href="https://agentskills.io">agentskills.io</a>.</td></tr>
<tr><td><b>Automações agendadas</b></td><td>Agendador cron integrado com entrega em qualquer plataforma. Relatórios diários, backups noturnos, auditorias semanais — tudo em linguagem natural, em execução autônoma.</td></tr>
<tr><td><b>Delega e paraleliza</b></td><td>Crie subagentes isolados para fluxos de trabalho paralelos. Escreva scripts Python que chamam ferramentas via RPC, colapsando pipelines de várias etapas em turnos sem custo de contexto.</td></tr>
<tr><td><b>Roda em qualquer lugar, não só no seu laptop</b></td><td>Sete backends de terminal — local, Docker, SSH, Singularity, Modal, Daytona e Vercel Sandbox. Daytona e Modal oferecem persistência serverless — o ambiente do seu agente hiberna quando ocioso e acorda sob demanda, custando quase nada entre as sessões. Rode em um VPS de US$ 5 ou em um cluster de GPUs.</td></tr>
<tr><td><b>Pronto para pesquisa</b></td><td>Geração de trajetórias em lote, compressão de trajetórias para treinar a próxima geração de modelos de chamada de ferramentas.</td></tr>
</table>

---

## Instalação rápida

### Linux, macOS, WSL2, Termux

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

### Windows (nativo, PowerShell)

> **Atenção:** No Windows nativo, o Hermes roda sem WSL — a CLI, o gateway, a TUI e as ferramentas funcionam nativamente. Se você preferir usar o WSL2, o comando de uma linha do Linux/macOS acima também funciona lá. Encontrou um bug? Por favor, [abra uma issue](https://github.com/NousResearch/hermes-agent/issues).

Execute isto no PowerShell:

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

O instalador cuida de tudo: uv, Python 3.11, Node.js, ripgrep, ffmpeg, **e um Git Bash portátil** (MinGit, descompactado em `%LOCALAPPDATA%\hermes\git` — sem necessidade de administrador, totalmente isolado de qualquer instalação de Git do sistema). O Hermes usa esse Git Bash embutido para executar comandos de shell.

Se você já tem o Git instalado, o instalador o detecta e o usa no lugar. Caso contrário, basta baixar os ~45 MB do MinGit — ele não toca nem interfere em nenhum Git do sistema.

> **Android / Termux:** O procedimento manual testado está documentado no [guia do Termux](https://hermes-agent.nousresearch.com/docs/getting-started/termux). No Termux, o Hermes instala o extra `.[termux]` curado porque o extra completo `.[all]` atualmente puxa dependências de voz incompatíveis com Android.
>
> **Windows:** O Windows nativo é totalmente suportado — o comando de uma linha do PowerShell acima instala tudo. Se você preferir usar o WSL2, o comando do Linux também funciona lá. A instalação nativa do Windows fica em `%LOCALAPPDATA%\hermes`; a do WSL2 fica em `~/.hermes`, como no Linux.

Depois da instalação:

```bash
source ~/.bashrc    # recarrega o shell (ou: source ~/.zshrc)
hermes              # comece a conversar!
```

### Solução de problemas

#### Windows Defender ou antivírus sinalizam o `uv.exe` como malware

Se o seu antivírus (Bitdefender, Windows Defender etc.) coloca em quarentena o `uv.exe` da pasta `bin` do Hermes (`%LOCALAPPDATA%\hermes\bin\uv.exe`), trata-se de um **falso positivo**. O arquivo é o `uv` da Astral — o gerenciador de pacotes Python escrito em Rust que o Hermes embute para gerenciar o próprio ambiente Python. Antivírus com detecção baseada em ML costumam sinalizar binários Rust não assinados que baixam e instalam pacotes.

**Para verificar se a sua cópia é autêntica:**

```powershell
# Instale a GitHub CLI se necessário
winget install --id GitHub.cli

# Faça login no GitHub
gh auth login

# Execute a verificação
$uv = "$env:LOCALAPPDATA\hermes\bin\uv.exe"
$ver = (& $uv --version).Split(' ')[1]
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$zip = "$env:TEMP\uv.zip"
Invoke-WebRequest "https://github.com/astral-sh/uv/releases/download/$ver/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip -UseBasicParsing
gh attestation verify $zip --repo astral-sh/uv
Expand-Archive $zip "$env:TEMP\uv_x" -Force
(Get-FileHash "$env:TEMP\uv_x\uv.exe").Hash -eq (Get-FileHash $uv).Hash
```

Se a verificação de atestação disser "Verification succeeded" e a última linha imprimir `True`, está tudo certo.

**Para colocar o Hermes na lista de exceções:**
- **Windows Defender:** Execute o PowerShell como Administrador → `Add-MpPreference -ExclusionPath "$env:LOCALAPPDATA\hermes\bin"`
- **Bitdefender:** Adicione uma exceção no console do Bitdefender (Protection > Antivirus > Settings > Manage Exceptions)
- Coloque a **pasta** na lista de exceções, não o hash do arquivo — o Hermes atualiza o `uv` e o hash muda a cada versão

Para mais contexto, veja os relatos da Astral no upstream: [astral-sh/uv#13553](https://github.com/astral-sh/uv/issues/13553), [astral-sh/uv#15011](https://github.com/astral-sh/uv/issues/15011), [astral-sh/uv#10079](https://github.com/astral-sh/uv/issues/10079).

---

## Primeiros passos

```bash
hermes              # CLI interativa — inicie uma conversa
hermes model        # Escolha seu provedor de LLM e modelo
hermes tools        # Configure quais ferramentas estão ativadas
hermes config set   # Defina valores individuais de configuração
hermes config get   # Exiba valores individuais de configuração
hermes gateway      # Inicie o gateway de mensageria (Telegram, Discord etc.)
hermes setup        # Execute o assistente de configuração completo (configura tudo de uma vez)
hermes claw migrate # Migre do OpenClaw (se estiver vindo do OpenClaw)
hermes update       # Atualize para a versão mais recente
hermes doctor       # Diagnostique eventuais problemas
```

📖 **[Documentação completa →](https://hermes-agent.nousresearch.com/docs/)**

---

## Pare de colecionar chaves de API — Nous Portal

O Hermes funciona com o provedor que você quiser — isso não vai mudar. Mas, se você prefere não colecionar cinco chaves de API separadas para modelo, busca na web, geração de imagens, TTS e navegador na nuvem, o **[Nous Portal](https://portal.nousresearch.com)** cobre todas elas sob uma única assinatura:

- **Mais de 300 modelos** — escolha qualquer um deles com `/model <name>`
- **Tool Gateway** — busca na web (Firecrawl), geração de imagens (FAL), texto para fala (OpenAI), navegador na nuvem (Browser Use), tudo roteado pela sua assinatura. Sem contas extras.

Um comando a partir de uma instalação nova:

```bash
hermes setup --portal
```

Isso faz o login via OAuth, define a Nous como seu provedor e ativa o Tool Gateway. Confira o que está configurado a qualquer momento com `hermes portal info`. Detalhes completos na [página de documentação do Tool Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway).

Você ainda pode usar as suas próprias chaves por ferramenta sempre que quiser — o gateway é por backend, não tudo-ou-nada.

---

## Referência rápida: CLI vs Mensageria

O Hermes tem dois pontos de entrada: inicie a interface de terminal com `hermes` ou rode o gateway e converse com ele pelo Telegram, Discord, Slack, WhatsApp, Signal ou E-mail. Uma vez dentro de uma conversa, muitos comandos de barra são compartilhados entre as duas interfaces.

| Ação                           | CLI                                           | Plataformas de mensageria                                                        |
| ------------------------------ | --------------------------------------------- | -------------------------------------------------------------------------------- |
| Começar a conversar            | `hermes`                                      | Rode `hermes gateway setup` + `hermes gateway start` e envie uma mensagem ao bot |
| Iniciar conversa do zero       | `/new` ou `/reset`                            | `/new` ou `/reset`                                                               |
| Trocar de modelo               | `/model [provider:model]`                     | `/model [provider:model]`                                                        |
| Definir uma personalidade      | `/personality [name]`                         | `/personality [name]`                                                            |
| Repetir ou desfazer o último turno | `/retry`, `/undo`                         | `/retry`, `/undo`                                                                |
| Comprimir contexto / ver uso   | `/compress`, `/usage`, `/insights [--days N]` | `/compress`, `/usage`, `/insights [days]`                                        |
| Navegar pelas skills           | `/skills` ou `/<skill-name>`                  | `/<skill-name>`                                                                  |
| Interromper o trabalho atual   | `Ctrl+C` ou envie uma nova mensagem           | `/stop` ou envie uma nova mensagem                                               |
| Status específico da plataforma | `/platforms`                                 | `/status`, `/sethome`                                                            |

Para as listas completas de comandos, veja o [guia da CLI](https://hermes-agent.nousresearch.com/docs/user-guide/cli) e o [guia do Gateway de Mensageria](https://hermes-agent.nousresearch.com/docs/user-guide/messaging).

---

## Documentação

Toda a documentação está em **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)**:

| Seção                                                                                               | Conteúdo                                                    |
| --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| [Início rápido](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart)              | Instalar → configurar → primeira conversa em 2 minutos      |
| [Uso da CLI](https://hermes-agent.nousresearch.com/docs/user-guide/cli)                             | Comandos, atalhos de teclado, personalidades, sessões        |
| [Configuração](https://hermes-agent.nousresearch.com/docs/user-guide/configuration)                 | Arquivo de configuração, provedores, modelos, todas as opções |
| [Gateway de Mensageria](https://hermes-agent.nousresearch.com/docs/user-guide/messaging)            | Telegram, Discord, Slack, WhatsApp, Signal, Home Assistant   |
| [Segurança](https://hermes-agent.nousresearch.com/docs/user-guide/security)                         | Aprovação de comandos, pareamento por DM, isolamento em contêiner |
| [Ferramentas e Toolsets](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools)      | Mais de 40 ferramentas, sistema de toolsets, backends de terminal |
| [Sistema de Skills](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills)          | Memória procedimental, Skills Hub, criação de skills         |
| [Memória](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory)                    | Memória persistente, perfis de usuário, boas práticas        |
| [Integração MCP](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp)                | Conecte qualquer servidor MCP para capacidades estendidas    |
| [Agendamento Cron](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron)             | Tarefas agendadas com entrega em plataformas                 |
| [Arquivos de Contexto](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files) | Contexto do projeto que molda cada conversa                  |
| [Arquitetura](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture)              | Estrutura do projeto, loop do agente, classes principais     |
| [Contribuir](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing)               | Configuração de desenvolvimento, processo de PR, estilo de código |
| [Referência da CLI](https://hermes-agent.nousresearch.com/docs/reference/cli-commands)              | Todos os comandos e flags                                    |
| [Variáveis de Ambiente](https://hermes-agent.nousresearch.com/docs/reference/environment-variables) | Referência completa de variáveis de ambiente                 |

---

## Migrando do OpenClaw

Se você está vindo do OpenClaw, o Hermes pode importar automaticamente suas configurações, memórias, skills e chaves de API.

**Durante a configuração inicial:** o assistente de configuração (`hermes setup`) detecta automaticamente o `~/.openclaw` e oferece a migração antes do início da configuração.

**A qualquer momento depois da instalação:**

```bash
hermes claw migrate              # Migração interativa (preset completo)
hermes claw migrate --dry-run    # Prévia do que seria migrado
hermes claw migrate --preset user-data   # Migre sem segredos
hermes claw migrate --overwrite  # Sobrescreva conflitos existentes
```

O que é importado:

- **SOUL.md** — arquivo de persona
- **Memórias** — entradas de MEMORY.md e USER.md
- **Skills** — skills criadas pelo usuário → `~/.hermes/skills/openclaw-imports/`
- **Allowlist (lista de permissões) de comandos** — padrões de aprovação
- **Configurações de mensageria** — configurações de plataformas, usuários permitidos, diretório de trabalho
- **Chaves de API** — segredos na allowlist (Telegram, OpenRouter, OpenAI, Anthropic, ElevenLabs)
- **Ativos de TTS** — arquivos de áudio do workspace
- **Instruções do workspace** — AGENTS.md (com `--workspace-target`)

Consulte `hermes claw migrate --help` para ver todas as opções, ou use a skill `openclaw-migration` para uma migração interativa guiada pelo agente, com prévias de dry-run.

---

## Contribuir

Contribuições são bem-vindas! Veja o [Guia de Contribuição](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) para configuração de desenvolvimento, estilo de código e processo de PR.

Início rápido para contribuidores — use o instalador padrão e trabalhe a partir do checkout git completo que ele cria em `$HERMES_HOME/hermes-agent` (geralmente `~/.hermes/hermes-agent`). Isso corresponde ao layout usado pelo `hermes update`, pelo venv gerenciado, pelas dependências lazy, pelo gateway e pelas ferramentas de documentação.

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
cd "${HERMES_HOME:-$HOME/.hermes}/hermes-agent"
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

Alternativa com clone manual (para clones descartáveis/CI, em que você intencionalmente não quer o layout de instalação gerenciado):

Crie o venv fora da árvore de código clonada — um venv dentro do diretório onde o agente opera pode ser apagado por um comando com caminho relativo executado pelo próprio agente contra o seu checkout, destruindo o runtime no meio da sessão.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ~/.hermes/venvs/hermes-dev --python 3.11
source ~/.hermes/venvs/hermes-dev/bin/activate
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

---

## Comunidade

- 💬 [Discord](https://discord.gg/NousResearch)
- 📚 [Skills Hub](https://agentskills.io)
- 🐛 [Issues](https://github.com/NousResearch/hermes-agent/issues)
- 🔌 [computer-use-linux](https://github.com/avifenesh/computer-use-linux) — Servidor MCP de controle de desktop Linux para o Hermes e outros hosts MCP, com árvores de acessibilidade AT-SPI, entrada Wayland/X11, capturas de tela e direcionamento de janelas do compositor.
- 🔌 [HermesClaw](https://github.com/AaronWong1999/hermesclaw) — Ponte comunitária para WeChat: rode o Hermes Agent e o OpenClaw na mesma conta de WeChat.

---

## Licença

MIT — veja [LICENSE](LICENSE).

Criado pela [Nous Research](https://nousresearch.com).
