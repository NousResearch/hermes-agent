"""生成 deploy/sql 下的 DDL：`python -m pdr_server.db.render`（在 server/ 目录执行）。

PostgreSQL 额外输出行级安全（RLS）策略，作为应用层租户/区域过滤之外的第二道防线。
"""

from __future__ import annotations

import sys
from pathlib import Path

from .naming import check_all
from .spec import render
from .tables import TABLES

SCHEMAS = ["src", "ods", "dwd", "dws", "ads", "dim"]
OUT = Path(__file__).resolve().parents[3] / "deploy" / "sql"
HEADER = "-- 由 server/pdr_server/db/tables.py 生成，请勿手改：python -m pdr_server.db.render\n"

# 区域过滤不适用的表：维度/权限配置本身、租户级汇总
_REGION_EXEMPT_SCHEMAS = {"dim"}


def _pg_rls(spec) -> list[str]:
    """租户隔离 + 区域隔离策略。会话变量由后端在每个事务开头 SET LOCAL：
    pdr.tenant_code      当前租户编码
    pdr.region_all       '1' 表示数据范围为本租户全部（跳过区域过滤）
    pdr.region_codes     逗号分隔的可见区域编码（已展开下级）
    平台租户（platform）的共享行对所有租户只读可见。
    """
    t = spec.full_name
    tenant = "tenant_code = current_setting('pdr.tenant_code', true)"
    shared = "tenant_code = 'platform'"
    stmts = [f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY", f"ALTER TABLE {t} FORCE ROW LEVEL SECURITY"]
    has_region = "region_code" in spec.column_names() and spec.schema not in _REGION_EXEMPT_SCHEMAS
    region = (
        " AND (current_setting('pdr.region_all', true) = '1' OR region_code = ''"
        " OR region_code = ANY (string_to_array(current_setting('pdr.region_codes', true), ',')))"
        if has_region else ""
    )
    stmts.append(f"DROP POLICY IF EXISTS p_tenant_read ON {t}")
    stmts.append(f"CREATE POLICY p_tenant_read ON {t} FOR SELECT USING (({tenant}{region}) OR {shared})")
    stmts.append(f"DROP POLICY IF EXISTS p_tenant_write ON {t}")
    stmts.append(
        f"CREATE POLICY p_tenant_write ON {t} FOR ALL USING ({tenant}{region}) WITH CHECK ({tenant}{region})"
    )
    return stmts


def build(dialect: str) -> str:
    out = [HEADER]
    if dialect == "postgres":
        out.append("CREATE EXTENSION IF NOT EXISTS vector;\n")
        out += [f"CREATE SCHEMA IF NOT EXISTS {s};\n" for s in SCHEMAS]
    else:
        out += [f"CREATE DATABASE IF NOT EXISTS {s} DEFAULT CHARSET utf8mb4 COLLATE utf8mb4_0900_ai_ci;\n" for s in SCHEMAS]
    for schema in SCHEMAS:
        out.append(f"\n-- {'=' * 70}\n-- {schema}\n-- {'=' * 70}\n")
        for spec in (t for t in TABLES if t.schema == schema):
            stmts = render(spec, dialect)  # type: ignore[arg-type]
            if dialect == "postgres":
                stmts += _pg_rls(spec)
            out.append("\n" + ";\n".join(stmts) + ";\n")
    return "".join(out)


def main() -> int:
    errors = check_all(TABLES)
    if errors:
        print("命名规范检查未通过：\n" + "\n".join(errors), file=sys.stderr)
        return 1
    for dialect, fname in (("postgres", "postgres/pdr_ddl.sql"), ("mysql", "mysql/pdr_ddl.sql")):
        path = OUT / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(build(dialect), encoding="utf-8")
        print(f"已生成 {path}（{len(TABLES)} 张表）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
