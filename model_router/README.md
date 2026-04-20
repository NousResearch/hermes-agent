# Model Router

ראוטר אופרטיבי לבחירת מודל לפי סוג משימה, עדיפות, פרטיות, מצב מכסה ומהירות — עם telemetry, feedback loop, ניתוח ביצועים, segmented suggestions והצעת patch ל-policy.

## Quick Start

```bash
cd model_router
python3 -m pip install -e .
validate-router-config router_config.yaml
python3 scripts/demo_workflow.py
```

אם יש סביבת test מלאה:

```bash
python3 -m pip install -e .[dev]
pytest -q
```

## מה כבר עובד בפועל

זה כבר לא scaffold בלבד. כרגע יש כאן מערכת עובדת עם החלקים הבאים:

- `model_router.py` — מנוע הראוטר
- `route_cli.py` — CLI לבחירת מודל
- `telemetry.py` — append-only JSONL logging ל-decisions
- `log_router_feedback.py` — append-only JSONL logging ל-feedback
- `analyze_telemetry.py` — summary + join לפי `request_id`
- `suggest_router_changes.py` — המלצות כלליות
- `suggest_router_changes_v2.py` — segmented suggestions
- `propose_config_patch.py` — הצעת patch ל-`router_config.yaml`
- `validate_router_config.py` — ולידציה לקונפיג

## מבנה הפרויקט

```txt
model_router/
  README.md
  pyproject.toml
  router_config.yaml
  model_router.py
  route_cli.py
  telemetry.py
  log_router_feedback.py
  analyze_telemetry.py
  suggest_router_changes.py
  suggest_router_changes_v2.py
  propose_config_patch.py
  validate_router_config.py
  Makefile
  justfile
  scripts/
    demo_workflow.py
  sample_data/
  tests/
```

## CLI commands זמינים

אחרי `pip install -e .` יהיו זמינים:

- `validate-router-config`
- `route-model`
- `log-router-feedback`
- `analyze-router-telemetry`
- `suggest-router-changes`
- `suggest-router-segments`
- `propose-router-patch`

בנוסף יש demo script מקומי:

- `python3 scripts/demo_workflow.py`

## דוגמאות שימוש

### 1. בחירת מודל
```bash
route-model --task-type coding --mode execute --priority high --has-code --json
```

### 2. כתיבת decision log
```bash
route-model \
  --task-type coding \
  --mode execute \
  --priority high \
  --has-code \
  --log-path logs/router-decisions.jsonl \
  --request-id req-1 \
  --json
```

### 3. כתיבת feedback
```bash
log-router-feedback \
  --log-path logs/router-decisions.jsonl \
  --request-id req-1 \
  --outcome success \
  --actual-model-used gpt-5.4 \
  --user-rating 5 \
  --notes "worked well"
```

### 4. ניתוח telemetry
```bash
analyze-router-telemetry logs/router-decisions.jsonl
```

### 5. הצעות כלליות
```bash
suggest-router-changes logs/router-decisions.jsonl
```

### 6. segmented suggestions
```bash
suggest-router-segments logs/router-decisions.jsonl
```

### 7. הצעת patch ל-config
```bash
propose-router-patch logs/router-decisions.jsonl --config router_config.yaml
```

### 8. הרצת demo end-to-end
```bash
python3 scripts/demo_workflow.py
```

הסקריפט:
- יוצר `sample_data/demo_router_log.jsonl`
- מדמה 3 החלטות + 3 אירועי feedback
- מריץ summary + suggestions + segmented suggestions
- בונה patch proposal
- מוודא שה-config המוצע עדיין valid
- מכוסה גם בטסט ייעודי: `tests/test_demo_workflow.py`

## `router_config.yaml`

הקובץ מגדיר כרגע:

- `router.default_model`
- `base_by_task`
- `fallbacks`
- `mode_overrides`
- `reviewers`
- `policy_overrides`

דוגמה:

```yaml
policy_overrides:
  - name: "auto-chat-medium-critical"
    when:
      task_type: "chat"
      priority: "medium"
      quota: "critical"
    force: "claude-sonnet-4.6"
    reason: "telemetry: avoid weak routing for chat+medium+critical"
```

## סדר הקדימויות בראוטר

הזרימה כרגע היא:

1. normalize
2. base selection
3. mode override
4. hard safety overrides
5. soft routing overrides
6. policy overrides
7. hard safety overrides שוב
8. constraints
9. reviewer

המשמעות:
- policy יכול לדרוס quota/speed heuristics
- אבל לא יכול לדרוס privacy safety

## Makefile / justfile

### Makefile
```bash
make install-dev
make validate-config
make test
make check
make analyze
make suggest
make suggest-segments
make propose-patch
make demo
make review
```

### justfile
```bash
just install-dev
just validate-config
just test
just check
just analyze
just suggest
just suggest-segments
just propose-patch
just demo
just review
```

## CI

יש workflow ב:

```txt
.github/workflows/router-ci.yml
```

הוא רץ על שינויים ב-`model_router/**` ומבצע:

1. install (`.[dev]`)
2. `validate-router-config router_config.yaml`
3. demo workflow end-to-end
4. `pytest -q`

## סטטוס נוכחי

עובד בפועל:
- routing
- policy overrides
- decision logging
- feedback logging
- telemetry analysis
- general suggestions
- segmented suggestions
- config patch proposal
- config validation
- demo workflow end-to-end
- demo workflow test coverage

עדיין חסר או תלוי סביבה:
- הרצת `pytest` מלאה בסביבה עם `pip/pytest`
- CI ירוק בפועל על runner אמיתי
- polish קטן נוסף אם תרצה, אבל לא חסר רכיב v1 קריטי

## workflow מומלץ

```bash
route-model ... --log-path logs/router-decisions.jsonl --request-id req-123
log-router-feedback --log-path logs/router-decisions.jsonl --request-id req-123 --outcome success --actual-model-used gpt-5.4
analyze-router-telemetry logs/router-decisions.jsonl
suggest-router-changes logs/router-decisions.jsonl
suggest-router-segments logs/router-decisions.jsonl
propose-router-patch logs/router-decisions.jsonl --config router_config.yaml
```

זה כבר v1 פנימי די רציני, לא רק skeleton.
