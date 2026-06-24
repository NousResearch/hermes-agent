# QA Regressão Ágora — t_b6432a0d

**URL testada:** http://127.0.0.1:9119/agora?cb=1782191352

**Resumo:** 11/12 critérios validados com sucesso.

## Resultados

- **mention_autocomplete_opens**: PASS
  - `options`: ["@all\nTodos", "@todos\nTodos", "@agora-backend\nIdle", "@agora-frontend\nWorking", "@agora-qa\nWorking", "@tech-lead-gpt\nReviewing", "@test-helper\nWorking"]
  - `screenshot`: "/home/felipi/.hermes/hermes-agent/agora-regressao-t_b6432a0d/01_mention_dropdown_at.png"
- **mention_filter**: PASS
  - `options`: ["@agora-frontend\nWorking"]
  - `screenshot`: "/home/felipi/.hermes/hermes-agent/agora-regressao-t_b6432a0d/02_mention_dropdown_fr.png"
- **mention_escape_closes**: PASS
- **mention_keyboard_insert**: PASS
  - `value`: "@todos "
  - `screenshot`: "/home/felipi/.hermes/hermes-agent/agora-regressao-t_b6432a0d/03_mention_inserted_todos.png"
- **mention_a11y**: PASS
  - `detail`: {"composerRole": "combobox"}
- **admin_panel_opens**: PASS
  - `screenshot`: "/home/felipi/.hermes/hermes-agent/agora-regressao-t_b6432a0d/04_admin_form.png"
- **admin_form_labels**: PASS
  - `labels`: ["Nome", "Slug", "Descrição"]
- **admin_create_valid**: PASS
  - `slug`: "qa-regressao-admin-1782191352"
  - `channel`: {"found": true, "active": true, "text": "QA Regressão Admin 1782191352\nCanal criado pela regressão QA", "aria": "Canal QA Regressão Admin 1782191352: Canal criado pela regressão QA"}
  - `screenshot`: "/home/felipi/.hermes/hermes-agent/agora-regressao-t_b6432a0d/05_admin_created_channel.png"
- **db_sanity**: PASS
  - `canonical`: "/home/felipi/.hermes/agora.db"
  - `canonical_id`: 15
- **admin_duplicate_slug**: PASS
  - `error`: "Já existe um canal com esse slug."
  - `screenshot`: "/home/felipi/.hermes/hermes-agent/agora-regressao-t_b6432a0d/06_admin_duplicate_error.png"
- **admin_invalid_slug**: FAIL
  - error_text=None
  - `screenshot`: "/home/felipi/.hermes/hermes-agent/agora-regressao-t_b6432a0d/07_admin_invalid_slug.png"
- **channel_aria_labels**: PASS
  - `labels`: [{"label": "Canal Praça: Conversa geral entre agentes e humanos.", "role": "tab"}, {"label": "Canal Planejamento: Discussões sobre próximos passos e estratégia.", "role": "tab"}, {"label": "Canal Decisões: Propostas e decisões formais.", "role": "tab"}, {"label": "Canal Incidentes: Bloqueios, erros e ações de recuperação.", "role": "tab"}, {"label": "Canal Profarma: Tópicos relacionados ao workspace profarma.dev/Aura.", "role": "tab"}]

## Screenshots

- `/home/felipi/.hermes/hermes-agent/agora-regressao-t_b6432a0d/01_mention_dropdown_at.png`
- `/home/felipi/.hermes/hermes-agent/agora-regressao-t_b6432a0d/02_mention_dropdown_fr.png`
- `/home/felipi/.hermes/hermes-agent/agora-regressao-t_b6432a0d/03_mention_inserted_todos.png`
- `/home/felipi/.hermes/hermes-agent/agora-regressao-t_b6432a0d/04_admin_form.png`
- `/home/felipi/.hermes/hermes-agent/agora-regressao-t_b6432a0d/05_admin_created_channel.png`
- `/home/felipi/.hermes/hermes-agent/agora-regressao-t_b6432a0d/06_admin_duplicate_error.png`
- `/home/felipi/.hermes/hermes-agent/agora-regressao-t_b6432a0d/07_admin_invalid_slug.png`
- `/home/felipi/.hermes/hermes-agent/agora-regressao-t_b6432a0d/newtab_screenshot.png`
- `/home/felipi/.hermes/hermes-agent/agora-regressao-t_b6432a0d/test_screenshot.png`
