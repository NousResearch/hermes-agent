# /project Command Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a `/project` slash command that tags sessions by project, optionally links a working directory, shows the project in the status bar, and exports conversations as Markdown to `~/.hermes/Projects/<name>/`.

**Architecture:**
- DB: add `projects` table + `project` column on `sessions`
- CLI: `/project` handler in `cli.py` (mirrors `/title` pattern)
- Status bar: append `project: <name>` fragment after `title:` fragment
- Exporter: new `hermes_cli/project_exporter.py` — converts DB messages to Markdown
- Config: optional `projects.obsidian_vault` + `projects.auto_export` in `config.yaml`

**Tech Stack:** Python, SQLite (hermes_state.py), Rich/prompt_toolkit (cli.py), PyYAML (config.py)

---

## Task 1: DB 스키마 — projects 테이블 추가

**Objective:** `projects` 테이블과 `sessions.project` 컬럼을 DB 마이그레이션으로 추가한다.

**Files:**
- Modify: `hermes_state.py` (SCHEMA SQL 블록 + `_migrate()` 메서드)

**현재 SCHEMA 위치:** `hermes_state.py` 37~100라인 `CREATE TABLE IF NOT EXISTS` 블록

**Step 1:** `SCHEMA` 문자열에 `projects` 테이블 정의 추가

```python
CREATE TABLE IF NOT EXISTS projects (
    name       TEXT PRIMARY KEY,
    work_dir   TEXT,
    description TEXT,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_projects_name ON projects(name);
```

`sessions` 테이블 정의에도 `project TEXT` 컬럼을 추가:
```sql
-- sessions 테이블 마지막 컬럼에 추가
project TEXT,
```

**Step 2:** `_migrate()` 메서드 안에 마이그레이션 구문 추가 (기존 `ALTER TABLE sessions ADD COLUMN title TEXT` 패턴 참고)

```python
# sessions 테이블에 project 컬럼 추가
try:
    cursor.execute('ALTER TABLE sessions ADD COLUMN "project" TEXT')
except Exception:
    pass

# projects 테이블 신규 생성
cursor.execute("""
    CREATE TABLE IF NOT EXISTS projects (
        name        TEXT PRIMARY KEY,
        work_dir    TEXT,
        description TEXT,
        created_at  REAL NOT NULL
    )
""")
cursor.execute(
    "CREATE INDEX IF NOT EXISTS idx_projects_name ON projects(name)"
)
```

**Step 3:** 검증

```bash
cd /Users/macmini/.hermes/hermes-agent
python3 -c "
from hermes_state import SessionDB
db = SessionDB('/tmp/test_project.db')
import sqlite3, json
con = sqlite3.connect('/tmp/test_project.db')
tables = con.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()
print('Tables:', tables)
cols = con.execute(\"PRAGMA table_info(sessions)\").fetchall()
print('sessions cols:', [c[1] for c in cols])
con.close()
"
```
Expected: `projects` 테이블 존재, `sessions` 컬럼 목록에 `project` 포함

**Step 4: Commit**
```bash
git add hermes_state.py
git commit -m "feat(db): add projects table and sessions.project column"
```

---

## Task 2: SessionDB — project CRUD 메서드 추가

**Objective:** `hermes_state.py`의 `SessionDB` 클래스에 project 관련 메서드 4개를 추가한다.

**Files:**
- Modify: `hermes_state.py` (SessionDB 클래스 끝 부분)

**추가할 메서드:**

```python
# ── Project CRUD ─────────────────────────────────────────────────────────────

def upsert_project(self, name: str, work_dir: str | None = None, description: str | None = None) -> None:
    """프로젝트를 생성하거나 work_dir/description을 업데이트한다."""
    import time
    with self._connect() as con:
        con.execute(
            """
            INSERT INTO projects (name, work_dir, description, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                work_dir    = COALESCE(excluded.work_dir, work_dir),
                description = COALESCE(excluded.description, description)
            """,
            (name, work_dir, description, time.time()),
        )

def get_project(self, name: str) -> dict | None:
    """프로젝트 정보를 반환한다. 없으면 None."""
    with self._connect() as con:
        row = con.execute(
            "SELECT name, work_dir, description, created_at FROM projects WHERE name = ?",
            (name,),
        ).fetchone()
    if row is None:
        return None
    return {"name": row[0], "work_dir": row[1], "description": row[2], "created_at": row[3]}

def list_projects(self) -> list[dict]:
    """등록된 프로젝트 목록을 created_at 내림차순으로 반환한다."""
    with self._connect() as con:
        rows = con.execute(
            "SELECT name, work_dir, description, created_at FROM projects ORDER BY created_at DESC"
        ).fetchall()
    return [{"name": r[0], "work_dir": r[1], "description": r[2], "created_at": r[3]} for r in rows]

def set_session_project(self, session_id: str, project_name: str | None) -> bool:
    """세션에 프로젝트 태그를 설정(또는 해제)한다."""
    with self._connect() as con:
        cur = con.execute(
            "UPDATE sessions SET project = ? WHERE id = ?",
            (project_name, session_id),
        )
    return cur.rowcount > 0

def get_sessions_by_project(self, project_name: str, limit: int = 50) -> list[dict]:
    """특정 프로젝트에 속한 세션 목록을 반환한다."""
    with self._connect() as con:
        rows = con.execute(
            """
            SELECT id, title, started_at, message_count
            FROM sessions
            WHERE project = ?
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (project_name, limit),
        ).fetchall()
    return [{"id": r[0], "title": r[1], "started_at": r[2], "message_count": r[3]} for r in rows]
```

**Step 2:** 검증

```bash
python3 -c "
from hermes_state import SessionDB
db = SessionDB('/tmp/test_project.db')
db.upsert_project('ConvLSTM', '/Users/macmini/Python/ConvLSTM')
print(db.get_project('ConvLSTM'))
print(db.list_projects())
"
```
Expected: `{'name': 'ConvLSTM', 'work_dir': '/Users/macmini/Python/ConvLSTM', ...}`

**Step 3: Commit**
```bash
git add hermes_state.py
git commit -m "feat(db): add project CRUD methods to SessionDB"
```

---

## Task 3: /project 커맨드 등록 (commands.py)

**Objective:** `COMMAND_REGISTRY`에 `project` 커맨드를 등록하여 자동완성, 도움말, gateway dispatch가 모두 인식하도록 한다.

**Files:**
- Modify: `hermes_cli/commands.py`

**추가 위치:** `CommandDef("title", ...)` 바로 아래

```python
CommandDef(
    "project",
    "Tag session with a project, link a directory, list projects, or export conversations",
    "Session",
    args_hint="[name [path] | list | save | off]",
    subcommands=("list", "save", "off"),
),
```

**Step 2:** 검증

```bash
python3 -c "
from hermes_cli.commands import COMMAND_REGISTRY
cmd = next(c for c in COMMAND_REGISTRY if c.name == 'project')
print(cmd)
"
```
Expected: CommandDef 출력 확인

**Step 3: Commit**
```bash
git add hermes_cli/commands.py
git commit -m "feat(commands): register /project slash command"
```

---

## Task 4: 대화 Markdown 익스포터 (project_exporter.py 신규)

**Objective:** 특정 세션의 대화를 Obsidian 호환 Markdown 파일로 저장하는 모듈을 만든다.

**Files:**
- Create: `hermes_cli/project_exporter.py`

```python
"""Export session conversations to Markdown for Obsidian / project notes."""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_state import SessionDB


def _safe_filename(s: str) -> str:
    """문자열을 파일명으로 안전하게 변환."""
    s = re.sub(r'[^\w\s\-]', '', s).strip()
    s = re.sub(r'\s+', '_', s)
    return s[:60] or "session"


def export_session(
    db: "SessionDB",
    session_id: str,
    project_name: str,
    base_dir: str | None = None,
    obsidian_vault: str | None = None,
) -> str:
    """
    세션 대화를 Markdown으로 저장한다.

    저장 경로:
      {base_dir}/Projects/{project_name}/YYYY-MM-DD/{session_id}_{title}.md

    obsidian_vault가 설정된 경우 해당 경로에도 동일 파일을 복사한다.

    Returns:
        저장된 파일의 절대 경로
    """
    if base_dir is None:
        base_dir = str(Path.home() / ".hermes")

    # ── 세션 메타 로드 ────────────────────────────────────────────────────────
    session = db.get_session(session_id)
    if session is None:
        raise ValueError(f"Session not found: {session_id}")

    title = (session.get("title") or "").strip()
    started_at = session.get("started_at") or time.time()
    date_str = time.strftime("%Y-%m-%d", time.localtime(started_at))

    # ── 메시지 로드 ───────────────────────────────────────────────────────────
    messages = db.get_messages(session_id)

    # ── 저장 경로 결정 ────────────────────────────────────────────────────────
    project_dir = Path(base_dir) / "Projects" / project_name / date_str
    project_dir.mkdir(parents=True, exist_ok=True)

    fname_parts = [session_id]
    if title:
        fname_parts.append(_safe_filename(title))
    filename = "_".join(fname_parts) + ".md"
    dest_path = project_dir / filename

    # ── Markdown 생성 ─────────────────────────────────────────────────────────
    lines: list[str] = []

    # YAML frontmatter (Obsidian 호환)
    lines.append("---")
    lines.append(f"session: {session_id}")
    if title:
        lines.append(f"title: {title}")
    lines.append(f"project: {project_name}")
    lines.append(f"date: {date_str}")
    lines.append(f"model: {session.get('model') or ''}")
    lines.append(f"message_count: {session.get('message_count') or len(messages)}")
    lines.append("---")
    lines.append("")

    if title:
        lines.append(f"# {title}")
    else:
        lines.append(f"# Session {session_id}")
    lines.append("")
    lines.append(f"> Project: **{project_name}**  |  Date: {date_str}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 대화 본문
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content") or ""

        if role == "user":
            lines.append(f"**🧑 User:**")
        elif role == "assistant":
            lines.append(f"**🤖 Assistant:**")
        elif role == "tool":
            tool_name = msg.get("tool_name") or "tool"
            lines.append(f"**🔧 Tool ({tool_name}):**")
        else:
            lines.append(f"**{role}:**")

        if content:
            lines.append("")
            lines.append(content)
        lines.append("")
        lines.append("---")
        lines.append("")

    md_text = "\n".join(lines)

    # ── 저장 ──────────────────────────────────────────────────────────────────
    dest_path.write_text(md_text, encoding="utf-8")

    # Obsidian vault 복사 (설정된 경우)
    if obsidian_vault:
        vault_dir = Path(obsidian_vault).expanduser() / "Hermes" / project_name / date_str
        vault_dir.mkdir(parents=True, exist_ok=True)
        (vault_dir / filename).write_text(md_text, encoding="utf-8")

    return str(dest_path)
```

**Step 2:** 검증

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from hermes_state import SessionDB
from hermes_cli.project_exporter import export_session
db = SessionDB('/tmp/test_project.db')
# 실제 state.db로 테스트
import os; db2 = SessionDB(os.path.expanduser('~/.hermes/state.db'))
sessions = db2.list_sessions(limit=1)
if sessions:
    s = sessions[0]
    path = export_session(db2, s['id'], 'TestProject', base_dir='/tmp')
    print('Exported to:', path)
    import subprocess; subprocess.run(['head', '-20', path])
"
```

**Step 3: Commit**
```bash
git add hermes_cli/project_exporter.py
git commit -m "feat(exporter): add project_exporter for Markdown conversation export"
```

---

## Task 5: config.yaml 옵션 추가 (config.py)

**Objective:** `projects.obsidian_vault`와 `projects.auto_export` 설정을 config에 추가한다.

**Files:**
- Modify: `hermes_cli/config.py`

**Step 1:** config 접근 헬퍼 추가 (기존 `get_config_value` 패턴 참고)

```python
# config.py 에 추가 (기존 헬퍼 함수들 근처)

def get_projects_config(cfg: dict) -> dict:
    """projects 설정 섹션을 반환한다. 없으면 빈 dict."""
    return cfg.get("projects") or {}

def get_obsidian_vault(cfg: dict) -> str | None:
    """Obsidian vault 경로를 반환한다. 설정 안 됐으면 None."""
    val = get_projects_config(cfg).get("obsidian_vault")
    if val:
        return os.path.expanduser(str(val))
    return None

def get_projects_auto_export(cfg: dict) -> bool:
    """세션 종료 시 자동 export 여부. 기본값 False."""
    return bool(get_projects_config(cfg).get("auto_export", False))
```

**Step 2:** `default_config()` 또는 주석 예시에 섹션 추가

기존 `default_config` dict 찾아서 아래 추가:
```python
"projects": {
    # "obsidian_vault": "~/Documents/ObsidianVault",
    # "auto_export": False,
},
```

**Step 3:** 검증

```bash
python3 -c "
from hermes_cli.config import get_obsidian_vault, get_projects_auto_export
cfg = {'projects': {'obsidian_vault': '~/Documents/Vault', 'auto_export': True}}
print(get_obsidian_vault(cfg))
print(get_projects_auto_export(cfg))
"
```

**Step 4: Commit**
```bash
git add hermes_cli/config.py
git commit -m "feat(config): add projects.obsidian_vault and projects.auto_export settings"
```

---

## Task 6: /project 핸들러 구현 (cli.py)

**Objective:** `/project` 커맨드의 모든 서브커맨드를 처리하는 핸들러를 `cli.py`에 추가한다.

**Files:**
- Modify: `cli.py`

**Step 1:** `_pending_project` 인스턴스 변수 추가

`__init__` 에서 `self._pending_title = None` 근처에 추가:
```python
self._pending_project: str | None = None  # /project 미리 설정 (세션 생성 전)
```

**Step 2:** `/project` 핸들러 추가

`elif canonical == "title":` 블록 바로 아래에 추가:

```python
elif canonical == "project":
    self._handle_project_command(cmd_original)
```

**Step 3:** `_handle_project_command` 메서드 추가 (클래스 메서드로)

```python
def _handle_project_command(self, cmd_original: str) -> None:
    """Handle /project [name [path] | list | save | off]"""
    parts = cmd_original.strip().split(maxsplit=2)
    # parts[0] = "project", parts[1] = subcommand or name, parts[2] = optional path

    sub = parts[1].lower() if len(parts) > 1 else ""

    # ── /project (인수 없음) — 현재 프로젝트 정보 표시 ──────────────────────
    if not sub:
        if self._session_db:
            session = self._session_db.get_session(self.session_id)
            proj = (session or {}).get("project") or self._pending_project
            if proj:
                info = self._session_db.get_project(proj)
                _cprint(f"  Project: {proj}")
                if info and info.get("work_dir"):
                    _cprint(f"  Work dir: {info['work_dir']}")
            else:
                _cprint("  No project set. Usage: /project <name>")
        return

    # ── /project list ────────────────────────────────────────────────────────
    if sub == "list":
        if self._session_db:
            projects = self._session_db.list_projects()
            if not projects:
                _cprint("  No projects registered yet.")
            else:
                _cprint("  Registered projects:")
                for p in projects:
                    wd = f"  ({p['work_dir']})" if p.get("work_dir") else ""
                    _cprint(f"    • {p['name']}{wd}")
        return

    # ── /project off — 현재 세션에서 프로젝트 해제 ──────────────────────────
    if sub == "off":
        self._pending_project = None
        if self._session_db and self._session_db.get_session(self.session_id):
            self._session_db.set_session_project(self.session_id, None)
        _cprint("  Project tag removed from this session.")
        return

    # ── /project save — 현재 세션을 Markdown으로 수동 저장 ──────────────────
    if sub == "save":
        session = self._session_db.get_session(self.session_id) if self._session_db else None
        proj = (session or {}).get("project") or self._pending_project
        if not proj:
            _cprint("  No project set. Use /project <name> first.")
            return
        try:
            from hermes_cli.project_exporter import export_session
            from hermes_cli.config import get_obsidian_vault
            vault = get_obsidian_vault(self.config) if hasattr(self, "config") else None
            path = export_session(
                self._session_db,
                self.session_id,
                proj,
                obsidian_vault=vault,
            )
            _cprint(f"  Saved to: {path}")
        except Exception as e:
            _cprint(f"  Export failed: {e}")
        return

    # ── /project <name> [path] — 프로젝트 설정 ──────────────────────────────
    project_name = parts[1]          # 원본 케이스 유지
    work_dir = parts[2] if len(parts) > 2 else None

    # work_dir 검증
    if work_dir:
        expanded = os.path.expanduser(work_dir)
        if not os.path.isdir(expanded):
            _cprint(f"  Warning: directory not found: {expanded}")
        else:
            work_dir = expanded

    # ~/.hermes/Projects/<name> 디렉토리 생성
    import pathlib
    proj_base = pathlib.Path.home() / ".hermes" / "Projects" / project_name
    proj_base.mkdir(parents=True, exist_ok=True)

    if self._session_db:
        self._session_db.upsert_project(project_name, work_dir=work_dir)
        if self._session_db.get_session(self.session_id):
            self._session_db.set_session_project(self.session_id, project_name)
            _cprint(f"  Project set: {project_name}")
        else:
            self._pending_project = project_name
            _cprint(f"  Project queued: {project_name} (will be saved on first message)")
        if work_dir:
            _cprint(f"  Linked directory: {work_dir}")
        _cprint(f"  Notes directory: {proj_base}")
    else:
        _cprint("  Session database not available.")
```

**Step 4:** `_pending_project`를 세션 생성 시 DB에 반영

`_pending_title`이 세션 생성 후 DB에 저장되는 곳을 찾아서 (`set_session_title` 호출부), 바로 아래에 추가:

```python
if self._pending_project:
    self._session_db.upsert_project(self._pending_project)
    self._session_db.set_session_project(self.session_id, self._pending_project)
    self._pending_project = None
```

**Step 5:** `auto_export` — 세션 종료 시 자동 저장

세션 종료 핸들러(`end_session` 또는 `_on_exit` 등)를 찾아서 아래 코드 추가:

```python
# auto_export 처리
try:
    from hermes_cli.config import get_projects_auto_export, get_obsidian_vault
    if get_projects_auto_export(self.config):
        session = self._session_db.get_session(self.session_id) if self._session_db else None
        proj = (session or {}).get("project") if session else None
        if proj:
            from hermes_cli.project_exporter import export_session
            vault = get_obsidian_vault(self.config)
            export_session(self._session_db, self.session_id, proj, obsidian_vault=vault)
except Exception:
    pass
```

**Step 6:** 검증

```bash
python3 -m py_compile cli.py && echo "Syntax OK"
```

**Step 7: Commit**
```bash
git add cli.py
git commit -m "feat(cli): add /project command handler with list/save/off subcommands"
```

---

## Task 7: 상태바에 project 표시

**Objective:** `_get_status_bar_snapshot()`, `_build_status_bar_text()`, `_get_status_bar_fragments()`에 `project_name` 표시를 추가한다. `/title` 표시 패턴과 동일한 방식.

**Files:**
- Modify: `cli.py` (상태바 관련 3개 함수)

**Step 1:** `_get_status_bar_snapshot()` — snapshot dict에 `project` 추가

```python
# session_title 읽는 코드 바로 아래에 추가
project_name = None
if self._session_db:
    session = self._session_db.get_session(self.session_id)
    if session and session.get("project"):
        project_name = session["project"]
if not project_name and self._pending_project:
    project_name = self._pending_project
snapshot["project_name"] = project_name
```

**Step 2:** `_build_status_bar_text()` — 텍스트 status bar에 project 추가

`title:` 표시 코드 바로 앞에 삽입 (project가 title보다 앞에 오도록):

```python
project_name = snapshot.get("project_name")
if project_name:
    parts.append(f"project: {project_name}")
```

**Step 3:** `_get_status_bar_fragments()` — 컬러 fragments에 project 추가

`title:` fragments 코드 바로 앞에 삽입:

```python
project_name = snapshot.get("project_name")
if project_name:
    fragments.append(("class:statusbar.project", f" project: {project_name}"))
```

**Step 4:** 검증

```bash
python3 -m py_compile cli.py && echo "Syntax OK"
```

**Step 5: Commit**
```bash
git add cli.py
git commit -m "feat(cli): show project name in status bar"
```

---

## Task 8: 통합 검증 및 PR 브랜치 정리

**Objective:** 전체 기능이 동작하는지 확인하고 PR용 브랜치를 준비한다.

**Step 1:** 통합 동작 확인 (실제 hermes 실행)

```
hermes 실행 후 순서대로 입력:
1. /project ConvLSTM /Users/macmini/Python/ConvLSTM
   → "Project set: ConvLSTM", "Linked directory: ..."
2. /project
   → "Project: ConvLSTM", "Work dir: ..."
3. /project list
   → "• ConvLSTM (/Users/macmini/Python/ConvLSTM)"
4. (메시지 1개 전송)
5. /project save
   → "Saved to: ~/.hermes/Projects/ConvLSTM/YYYY-MM-DD/..."
6. 저장된 파일 열어서 YAML frontmatter + 대화 내용 확인
7. 상태바 확인: "project: ConvLSTM" 표시
8. /project off
   → "Project tag removed"
```

**Step 2:** 파일 생성 확인

```bash
ls ~/.hermes/Projects/ConvLSTM/
```

**Step 3:** git log 확인

```bash
git log --oneline | head -8
```

Expected:
```
feat(cli): show project name in status bar
feat(cli): add /project command handler ...
feat(config): add projects settings
feat(exporter): add project_exporter ...
feat(commands): register /project slash command
feat(db): add project CRUD methods to SessionDB
feat(db): add projects table and sessions.project column
```

**Step 4:** PR 브랜치 생성 (status bar title PR과 동일 패턴)

```bash
git fetch upstream
git checkout upstream/main -b feat/project-command
git cherry-pick <task1 commit> <task2 commit> ... <task7 commit>
python3 -m py_compile cli.py && echo OK
TOKEN=$(gh auth token)
git push "https://spegas:${TOKEN}@github.com/spegas/hermes-agent.git" feat/project-command
```

---

## 구현 순서 요약

```
Task 1  →  Task 2  →  Task 3  →  Task 4  →  Task 5  →  Task 6  →  Task 7  →  Task 8
  DB          DB       commands   exporter    config      cli         statusbar   PR
스키마       CRUD       등록       신규파일    설정 추가   핸들러      표시        정리
```

각 Task는 독립적으로 검증 가능하고, 앞 Task가 완료된 후 다음 Task를 진행한다.
Task 4(exporter)와 Task 5(config)는 서로 의존성이 없어 병렬 진행 가능.

---

## 파일 변경 요약

| 파일 | 변경 유형 | 내용 |
|---|---|---|
| `hermes_state.py` | Modify | projects 테이블, sessions.project 컬럼, CRUD 메서드 |
| `hermes_cli/commands.py` | Modify | /project CommandDef 등록 |
| `hermes_cli/project_exporter.py` | **Create** | Markdown export 모듈 |
| `hermes_cli/config.py` | Modify | obsidian_vault, auto_export 설정 |
| `cli.py` | Modify | /project 핸들러, 상태바 표시, _pending_project |
