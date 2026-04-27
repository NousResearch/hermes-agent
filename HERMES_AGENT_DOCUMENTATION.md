# Documentação Completa da Arquitetura e Funcionamento do Hermes Agent

## 1. Visão Geral
O **Hermes Agent** é um sistema de agente de Inteligência Artificial sofisticado e modular. Ele foi projetado para atuar em múltiplas frentes de comunicação (interfaces CLI, Gateway para mensageria como Discord e Slack, adaptadores de protocolo ACP), mantendo uma arquitetura central de tomada de decisão, persistência de estado e uso dinâmico de ferramentas ("tools") e habilidades ("skills").

## 2. Componentes Principais

### 2.1 Núcleo do Agente (`run_agent.py` e diretório `agent/`)
O coração do sistema reside no arquivo `run_agent.py`, onde está definida a classe `AIAgent`.
- **AIAgent**: Classe principal responsável por instanciar e gerenciar o loop de conversação do agente (`run_conversation`).
- **Loop de Execução**: O processo de receber inputs, enviar para o modelo, processar chamadas de funções (`_execute_tool_calls`) e retornar os resultados.
- **`agent/` (Pacote Core)**: Contém lógicas cruciais como:
  - `memory_manager.py`: Gerencia a memória de curto e longo prazo e contexto de sessão.
  - `prompt_builder.py`: Constrói dinamicamente os system prompts baseando-se no arquivo de perfil (ex: `SOUL.md`) para definir a identidade do agente.
  - `context_compressor.py`: Responsável por comprimir e otimizar o uso da janela de contexto para evitar desperdício de tokens.

### 2.2 Interfaces de Entrada e Interação
O sistema pode ser consumido através de diferentes interfaces:
- **Interface Terminal (CLI)** (`cli.py` e diretório `hermes_cli/`): Ponto de entrada interativo para usuários via terminal, construído com suporte a `prompt_toolkit`.
- **Gateway de Plataformas** (`gateway/`): Responsável por integrar o agente a plataformas externas. O `run.py` inicia o `GatewayRunner`, lidando com as sessões remotas de mensageria.
- **Agent Client Protocol (ACP)** (`acp_adapter/`): Uma implementação padronizada para interligar clientes de agentes, servindo como uma interface agnóstica (HTTP/Sockets) para expor as capacidades do agente para outros sistemas e IDEs.

### 2.3 Ferramentas e Habilidades (Tools & Skills)
O poder de execução do Hermes Agent vem da sua estrutura de ferramentas, dividida em implementações atômicas e rotinas complexas.
- **Registro de Ferramentas** (`tools/` e `model_tools.py`): Onde todas as ferramentas atômicas do agente são catalogadas. A orquestração (ex. `handle_function_call`) interliga a saída do modelo de IA diretamente à execução da função mapeada.
- **Ferramentas Atômicas** (`tools/`): Scripts que implementam ações específicas (ex: busca na web, manipulação de arquivos, execução de shell).
- **Habilidades (Skills)** (`skills/`): Compostos lógicos de alto nível ou rotinas que orquestram várias ferramentas atômicas para objetivos mais difíceis.

### 2.4 Estado e Persistência (`hermes_state.py`)
Para manter a continuidade de contexto entre reboots e sessões, o estado do agente é salvo localmente.
- **SQLite Database**: A classe `SessionState` contida em `hermes_state.py` fornece os schemas de dados garantindo o armazenamento, a rastreabilidade do histórico de conversas do agente e do seu estado em disco.

### 2.5 Ambientes de Execução Especializados (`environments/`)
Esse diretório disponibiliza "ambientes" padronizados em torno do agente, essenciais para testes de Benchmarks e Aprendizado por Reforço (RL).
- `agent_loop.py`: Contém o `HermesAgentLoop`, disponibilizando um loop encapsulado e programável.

### 2.6 Tarefas em Segundo Plano (`cron/`)
- Módulos para o agendamento (`scheduler.py` e `jobs.py`), permitindo ao agente executar ações repetitivas ou baseadas em gatilhos temporais de forma assíncrona.

## 3. Fluxo de Execução

1. **Inicialização**: A interface escolhida (CLI, Gateway ou Adapter) é inicializada.
2. **Setup do Agente**: A classe `AIAgent` é instanciada. O `prompt_builder.py` consolida as instruções fundamentais e lista de ferramentas.
3. **Loop Principal de Decisão**:
   - Um prompt do usuário é recebido.
   - O histórico de contexto é recuperado (`memory_manager.py`) e comprimido.
   - A requisição é despachada para o LLM.
4. **Resolução de Tool Calls**:
   - Se a IA invoca uma ferramenta, o orquestrador (`model_tools.py`) a intercepta, processa localmente a rotina pedida e reporta os achados ao LLM no mesmo loop, que pode continuar avaliando o resultado.
5. **Finalização e Resposta**: 
   - A resposta final texto/UI é entregue de volta ao usuário através da interface originária. O estado é salvo para acesso futuro.

## 4. Conclusão
O modelo arquitetural do Hermes Agent consolida uma divisão exemplar de camadas: o *cérebro* da aplicação (`agent/`, `run_agent.py`), as *interfaces* flexíveis (`gateway`, `cli`, `acp_adapter`) e os *membros atuadores* (`tools/`, `skills/`). Essa topologia propicia grande flexibilidade tanto para interações rotineiras na máquina local, quanto para a automação de workflows complexos em plataformas multiusuário.