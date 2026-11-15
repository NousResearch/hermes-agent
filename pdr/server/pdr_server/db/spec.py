"""表结构描述（TableSpec）与 DDL 渲染：同一份描述可渲染 PostgreSQL / MySQL 两种方言。

平台内所有表（平台元数据、数仓分层、业务源表）以及「DDL 设计器」生成的表都走这里，
因此命名规范检查（naming.py）只需对 TableSpec 做一次。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

Dialect = Literal["postgres", "mysql"]

# 时间粒度后缀 → 说明；需要 travel_date 分区的后缀
GRANULARITY_SUFFIXES: dict[str, str] = {
    "_hrs_f": "小时级全量快照", "_hrs": "小时级增量",
    "_wk_f": "周级全量快照", "_wk": "周级增量",
    "_mo_f": "月级全量快照", "_mo": "月级增量",
    "_q_f": "季度级全量快照", "_q": "季度级增量",
    "_yr_f": "年级全量快照", "_yr": "年级增量",
    "_di": "天级增量", "_df": "天级全量",
    "_zip": "拉链表", "_f": "非分区全量表",
}
NON_PARTITIONED_SUFFIXES = {"_f", "_zip"}
PARTITION_COLUMN = "travel_date"


def table_suffix(table: str) -> str | None:
    """最长匹配的粒度后缀（_hrs_f 优先于 _f）。"""
    for suffix in sorted(GRANULARITY_SUFFIXES, key=len, reverse=True):
        if table.endswith(suffix):
            return suffix
    return None


@dataclass
class Col:
    name: str
    type: str  # bigint/int/smallint/varchar(n)/char(n)/text/decimal(p,s)/date/timestamp/json/bool
    comment: str
    nullable: bool = True
    default: str | None = None  # 原样 SQL 表达式，'now' 表示当前时间
    auto: bool = False  # 自增主键


@dataclass
class TableSpec:
    schema: str
    name: str
    comment: str
    columns: list[Col]
    pk: list[str] = field(default_factory=lambda: ["id"])
    indexes: list[list[str]] = field(default_factory=list)
    uniques: list[list[str]] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        return f"{self.schema}.{self.name}"

    @property
    def partitioned(self) -> bool:
        suffix = table_suffix(self.name)
        return suffix is not None and suffix not in NON_PARTITIONED_SUFFIXES

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def effective_pk(self) -> list[str]:
        # 分区表的主键/唯一键必须包含分区列（PG 与 MySQL 都有此要求）
        if self.partitioned and self.pk and PARTITION_COLUMN not in self.pk:
            return [*self.pk, PARTITION_COLUMN]
        return self.pk

    def effective_uniques(self) -> list[list[str]]:
        if not self.partitioned:
            return self.uniques
        return [u if PARTITION_COLUMN in u else [*u, PARTITION_COLUMN] for u in self.uniques]


# ---------------------------------------------------------------------------
# 通用字段
# ---------------------------------------------------------------------------

def c_id() -> Col:
    return Col("id", "bigint", "自增主键", nullable=False, auto=True)


def c_tenant() -> Col:
    return Col("tenant_code", "varchar(32)", "租户编码（SaaS 租户隔离）", nullable=False)


def c_region() -> Col:
    return Col("region_code", "varchar(32)", "区域编码（区域权限隔离单元，关联 dim.dim_cus_region_f）", nullable=False, default="''")


def c_travel_date() -> Col:
    return Col(PARTITION_COLUMN, "char(8)", "分区日期，格式YYYYMMDD", nullable=False)


def c_audit() -> list[Col]:
    return [
        Col("record_status", "smallint", "记录状态：0-正常 1-已删除（逻辑删除）", nullable=False, default="0"),
        Col("create_user_code", "varchar(64)", "创建人用户编码，系统任务为 system", nullable=False, default="'system'"),
        Col("create_time", "timestamp", "创建时间", nullable=False, default="now"),
        Col("update_user_code", "varchar(64)", "最后更新人用户编码", nullable=False, default="'system'"),
        Col("update_time", "timestamp", "最后更新时间", nullable=False, default="now"),
    ]


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

_DECIMAL = re.compile(r"decimal\((\d+),\s*(\d+)\)")


def _pg_type(col: Col) -> str:
    t = col.type.lower()
    if col.auto:
        return "bigserial" if t == "bigint" else "serial"
    return {"timestamp": "timestamp(3)", "json": "jsonb", "bool": "boolean"}.get(t, t)  # vector → pgvector


def _mysql_type(col: Col) -> str:
    t = col.type.lower()
    if t == "timestamp":
        return "datetime(3)"
    if t == "bool":
        return "tinyint(1)"
    if t == "text":
        return "longtext"
    if t == "vector":  # MySQL 无向量类型：以 JSON 数组存储，不参与检索
        return "json"
    return t


def _default(col: Col, dialect: Dialect) -> str:
    if col.default is None:
        return ""
    if col.default == "now":
        return " DEFAULT CURRENT_TIMESTAMP" + ("(3)" if dialect == "mysql" else "")
    return f" DEFAULT {col.default}"


def _q(text: str) -> str:
    return text.replace("'", "''")


def render_pg(spec: TableSpec) -> list[str]:
    lines = []
    for col in spec.columns:
        null = "" if col.nullable else " NOT NULL"
        lines.append(f"  {col.name} {_pg_type(col)}{null}{_default(col, 'postgres')}")
    pk = spec.effective_pk()
    if pk:
        lines.append(f"  PRIMARY KEY ({', '.join(pk)})")
    for unique in spec.effective_uniques():
        lines.append(f"  UNIQUE ({', '.join(unique)})")
    tail = f" PARTITION BY RANGE ({PARTITION_COLUMN})" if spec.partitioned else ""
    stmts = [f"CREATE TABLE IF NOT EXISTS {spec.full_name} (\n" + ",\n".join(lines) + f"\n){tail}"]
    if spec.partitioned:
        stmts.append(
            f"CREATE TABLE IF NOT EXISTS {spec.full_name}_pdefault PARTITION OF {spec.full_name} DEFAULT"
        )
    stmts.append(f"COMMENT ON TABLE {spec.full_name} IS '{_q(spec.comment)}'")
    for col in spec.columns:
        stmts.append(f"COMMENT ON COLUMN {spec.full_name}.{col.name} IS '{_q(col.comment)}'")
    for i, idx in enumerate(spec.indexes):
        stmts.append(
            f"CREATE INDEX IF NOT EXISTS idx_{spec.name}_{i} ON {spec.full_name} ({', '.join(idx)})"
        )
    return stmts


def render_mysql(spec: TableSpec) -> list[str]:
    lines = []
    for col in spec.columns:
        null = "" if col.nullable else " NOT NULL"
        auto = " AUTO_INCREMENT" if col.auto else ""
        lines.append(
            f"  `{col.name}` {_mysql_type(col)}{null}{auto}{_default(col, 'mysql')} COMMENT '{_q(col.comment)}'"
        )
    pk = spec.effective_pk()
    if pk:
        lines.append(f"  PRIMARY KEY ({', '.join(pk)})")
    for i, unique in enumerate(spec.effective_uniques()):
        lines.append(f"  UNIQUE KEY uk_{spec.name}_{i} ({', '.join(unique)})")
    for i, idx in enumerate(spec.indexes):
        lines.append(f"  KEY idx_{spec.name}_{i} ({', '.join(idx)})")
    tail = (
        f"\nPARTITION BY RANGE COLUMNS({PARTITION_COLUMN}) (PARTITION p_max VALUES LESS THAN (MAXVALUE))"
        if spec.partitioned else ""
    )
    return [
        f"CREATE TABLE IF NOT EXISTS {spec.full_name} (\n" + ",\n".join(lines)
        + f"\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='{_q(spec.comment)}'{tail}"
    ]


def render(spec: TableSpec, dialect: Dialect) -> list[str]:
    return render_pg(spec) if dialect == "postgres" else render_mysql(spec)


def render_script(spec: TableSpec, dialect: Dialect) -> str:
    return ";\n".join(render(spec, dialect)) + ";\n"
