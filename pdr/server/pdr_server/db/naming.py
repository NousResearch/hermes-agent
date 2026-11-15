"""数据库命名规范检查（需求第 4 节的机器可执行版本）。

平台内置表与「DDL 设计器」提交的表都过同一套检查；返回中文错误列表，空列表表示通过。
"""

from __future__ import annotations

import re

from .spec import GRANULARITY_SUFFIXES, PARTITION_COLUMN, TableSpec, table_suffix

LAYERS = {"ods", "dwd", "dws", "ads", "dim", "src"}
DOMAINS = {"prd", "ord", "cus", "sup", "inv", "fin", "opr", "trf", "mkt"}
SNAKE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")
# 「取值含义」写法：0-成功 / vat_special-增值税专票；或声明为开放取值
ENUM_DOC = re.compile(r"(\b[\w.*]+-\S)|开放取值")

# 关键字后缀 → 允许的类型前缀
KEYWORD_TYPES: dict[str, tuple[str, ...]] = {
    "_date": ("date",),
    "_time": ("timestamp",),
    "_amt": ("decimal",),
    "_qty": ("int", "bigint", "smallint", "decimal"),
    "_rate": ("decimal",),
}
# 常见拼音片段（禁止拼音；仅做启发式拦截）
PINYIN = re.compile(
    r"(^|_)(jine|shuliang|riqi|shijian|mingcheng|bianhao|zhuangtai|leixing|kehu|dingdan|"
    r"gongyingshang|kucun|caiwu|jiesuan|fapiao|lvke|jingdian|jiage|yonghu|quyu)(_|$)"
)


def check_table(spec: TableSpec) -> list[str]:
    errors: list[str] = []
    name = spec.name
    if not SNAKE.match(name):
        errors.append(f"{spec.full_name}: 表名必须为小写下划线风格")
    parts = name.split("_")
    if parts[0] not in LAYERS:
        errors.append(f"{spec.full_name}: 表名须以分层前缀开头 {sorted(LAYERS)}")
    elif spec.schema != parts[0]:
        errors.append(f"{spec.full_name}: 库名须与分层一致（{parts[0]}）")
    if len(parts) < 3 or parts[1] not in DOMAINS:
        errors.append(f"{spec.full_name}: 第二段须为主题域前缀 {sorted(DOMAINS)}")
    if table_suffix(name) is None:
        errors.append(f"{spec.full_name}: 表名须以时间粒度后缀结尾 {list(GRANULARITY_SUFFIXES)}")
    if PINYIN.search(name):
        errors.append(f"{spec.full_name}: 表名疑似含拼音")
    if not spec.comment.strip():
        errors.append(f"{spec.full_name}: 缺少表注释")
    if spec.partitioned and PARTITION_COLUMN not in spec.column_names():
        errors.append(f"{spec.full_name}: 分区表必须包含分区字段 {PARTITION_COLUMN}")
    if not spec.partitioned and PARTITION_COLUMN in spec.column_names():
        errors.append(f"{spec.full_name}: 非分区表（_f/_zip）不应包含 {PARTITION_COLUMN}")

    seen: set[str] = set()
    for col in spec.columns:
        where = f"{spec.full_name}.{col.name}"
        if col.name in seen:
            errors.append(f"{where}: 字段重复")
        seen.add(col.name)
        if not SNAKE.match(col.name):
            errors.append(f"{where}: 字段名必须为小写下划线风格")
        if PINYIN.search(col.name):
            errors.append(f"{where}: 字段名疑似含拼音")
        if not col.comment.strip():
            errors.append(f"{where}: 缺少字段注释")
        if col.name.endswith(("_type", "_status")) and not ENUM_DOC.search(col.comment):
            errors.append(f"{where}: 类型/状态字段注释须写明每个取值含义（如 0-成功 1-失败）")
        if col.name == PARTITION_COLUMN:
            if col.type != "char(8)":
                errors.append(f"{where}: 分区字段类型须为 char(8)，格式 YYYYMMDD")
            continue
        for suffix, allowed in KEYWORD_TYPES.items():
            if col.name.endswith(suffix) and not col.type.lower().startswith(allowed):
                errors.append(f"{where}: 以 {suffix} 结尾的字段类型须为 {'/'.join(allowed)}，当前 {col.type}")
    return errors


def check_all(specs: list[TableSpec]) -> list[str]:
    errors: list[str] = []
    names: set[str] = set()
    for spec in specs:
        if spec.full_name in names:
            errors.append(f"{spec.full_name}: 表重复定义")
        names.add(spec.full_name)
        errors.extend(check_table(spec))
    return errors
