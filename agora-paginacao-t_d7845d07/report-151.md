# QA Paginação Ágora — t_d7845d07

**Data:** 2026-06-22T21:38:32.640647Z
**Resultado geral:** REPROVADO
**Mensagens seed:** 150
**PAGE_SIZE:** 50
**Cargas de histórico:** 0
**Contagem final:** None

## Passos

### ✅ list channels
```json
{
  "channels": [
    {
      "text": "Praça",
      "selected": true
    },
    {
      "text": "Planejamento",
      "selected": false
    },
    {
      "text": "Decisões",
      "selected": false
    },
    {
      "text": "Incidentes",
      "selected": false
    },
    {
      "text": "Profarma",
      "selected": false
    },
    {
      "text": "QA Paginação",
      "selected": false
    }
  ]
}
```

### ✅ select qa-paginacao
```json
{
  "selection": {
    "clicked": true,
    "text": "QA Paginação"
  }
}
```

### ✅ initial load
```json
{
  "state": {
    "scrollTop": 2918,
    "scrollHeight": 3133,
    "clientHeight": 215,
    "messageCount": 50,
    "firstText": "msg-101 — mensagem de teste 102/151",
    "lastText": "msg-150 — mensagem de teste 151/151",
    "loadingOlder": false,
    "feedEndVisible": false,
    "feedEndText": null
  }
}
```

### ✅ initial count == 50
```json
{
  "count": 50
}
```

### ✅ older load 1
```json
{
  "before_count": 50,
  "after_count": 100,
  "firstText": "msg-051 — mensagem de teste 52/151",
  "lastText": "msg-150 — mensagem de teste 151/151",
  "scrollTop": 3133,
  "feedEndVisible": false,
  "feedEndText": null
}
```

### ✅ older load 2
```json
{
  "before_count": 100,
  "after_count": 150,
  "firstText": "msg-001 — mensagem de teste 2/151",
  "lastText": "msg-150 — mensagem de teste 151/151",
  "scrollTop": 3132,
  "feedEndVisible": false,
  "feedEndText": null
}
```

### ✅ older load 3
```json
{
  "before_count": 150,
  "after_count": 151,
  "firstText": "msg-000 — mensagem de teste 1/151",
  "lastText": "msg-150 — mensagem de teste 151/151",
  "scrollTop": 102,
  "feedEndVisible": true,
  "feedEndText": "Início da conversa"
}
```

### ❌ exception
**Erro:** carregou mais mensagens do que o seed (151)
```json
{}
```


## Screenshots

- /home/felipi/.hermes/hermes-agent/agora-paginacao-t_d7845d07/screenshots/01_initial_load.png
- /home/felipi/.hermes/hermes-agent/agora-paginacao-t_d7845d07/screenshots/02_older_load_01.png
- /home/felipi/.hermes/hermes-agent/agora-paginacao-t_d7845d07/screenshots/02_older_load_02.png
- /home/felipi/.hermes/hermes-agent/agora-paginacao-t_d7845d07/screenshots/02_older_load_03.png
- /home/felipi/.hermes/hermes-agent/agora-paginacao-t_d7845d07/screenshots/03_final_state.png
- /home/felipi/.hermes/hermes-agent/agora-paginacao-t_d7845d07/screenshots/99_exception.png