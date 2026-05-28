"""SQL guard for read-only RecruitmentSystem queries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .config import QuerySettings, TableMapping
from .models import GuardResult

_DANGEROUS_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|replace|grant|revoke|merge|call|execute|load_file|sleep|benchmark)\b",
    re.IGNORECASE,
)
_COMMENT_RE = re.compile(r"(--|#|/\*|\*/)")
_TABLE_RE = re.compile(r"\b(?:from|join)\s+((?:`?[A-Za-z_][A-Za-z0-9_]*`?\.)?`?[A-Za-z_][A-Za-z0-9_]*`?)", re.IGNORECASE)
_LIMIT_RE = re.compile(r"\blimit\s+(\d+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class SQLGuard:
    mapping: TableMapping
    settings: QuerySettings

    def validate(
        self,
        sql: str,
        *,
        params: dict[str, Any] | None = None,
        current_user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> GuardResult:
        params = params or {}
        raw = (sql or "").strip()
        if not raw:
            return GuardResult(False, raw, "SQL 不能为空")
        if ";" in raw:
            return GuardResult(False, raw, "禁止执行多语句 SQL")
        if _COMMENT_RE.search(raw):
            return GuardResult(False, raw, "禁止使用 SQL 注释")
        if not re.match(r"^\s*select\b", raw, re.IGNORECASE):
            return GuardResult(False, raw, "只允许 SELECT 查询")
        if _DANGEROUS_RE.search(raw):
            return GuardResult(False, raw, "SQL 包含危险关键字")

        tables = _extract_tables(raw)
        allowed_tables = self.mapping.actual_tables()
        if not tables:
            return GuardResult(False, raw, "未识别到可查询表")
        unauthorized_tables = tables - allowed_tables
        if unauthorized_tables:
            return GuardResult(False, raw, f"禁止访问未授权表: {sorted(unauthorized_tables)}")

        selected_fields = _extract_selected_fields(raw)
        if "*" in selected_fields:
            return GuardResult(False, raw, "禁止 SELECT *，必须显式选择授权字段")
        allowed_fields = set()
        fields_by_table = self.mapping.allowed_fields_by_table()
        for table in tables:
            allowed_fields.update(fields_by_table.get(table, set()))
        unauthorized_fields = selected_fields - allowed_fields
        if unauthorized_fields:
            return GuardResult(False, raw, f"禁止访问未授权字段: {sorted(unauthorized_fields)}")

        normalized = _normalize_for_conditions(raw)
        if self.mapping.attendance_table in tables:
            user_field = self.mapping.require_attendance_field("user_id")
            if not _has_bound_condition(normalized, user_field, ("current_user_id", "user_id")):
                return GuardResult(False, raw, "考勤查询必须限定当前 user_id")
            if current_user_id is not None:
                bound_value = params.get("current_user_id", params.get("user_id"))
                if bound_value is not None and str(bound_value) != str(current_user_id):
                    return GuardResult(False, raw, "考勤查询 user_id 与当前用户不一致")

        for table in tables:
            tenant_field = _tenant_field_for_table(self.mapping, table)
            if tenant_id and tenant_field and not _has_bound_condition(normalized, tenant_field, ("tenant_id",)):
                return GuardResult(False, raw, "多租户查询必须限定 tenant_id")

        limited_sql, applied_limit = self._ensure_limit(raw)
        return GuardResult(True, limited_sql, None, applied_limit)

    def _ensure_limit(self, sql: str) -> tuple[str, int]:
        max_limit = max(1, self.settings.max_limit)
        default_limit = min(max(1, self.settings.default_limit), max_limit)
        match = _LIMIT_RE.search(sql)
        if not match:
            return f"{sql} LIMIT {default_limit}", default_limit

        current = int(match.group(1))
        if current <= max_limit:
            return sql, current
        limited_sql = f"{sql[:match.start(1)]}{max_limit}{sql[match.end(1):]}"
        return limited_sql, max_limit


def _extract_tables(sql: str) -> set[str]:
    return {_strip_identifier(match.group(1)) for match in _TABLE_RE.finditer(sql)}


def _extract_selected_fields(sql: str) -> set[str]:
    match = re.search(r"\bselect\b(.+?)\bfrom\b", sql, re.IGNORECASE | re.DOTALL)
    if not match:
        return set()
    fields = set()
    for expr in _split_csv(match.group(1)):
        clean = expr.strip()
        if not clean:
            continue
        if clean == "*":
            fields.add("*")
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*\(", clean):
            continue
        clean = re.split(r"\s+as\s+", clean, flags=re.IGNORECASE)[0]
        clean = clean.split()[0]
        fields.add(_strip_identifier(clean))
    return fields


def _split_csv(value: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in value:
        if char == "(":
            depth += 1
        elif char == ")" and depth:
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current))
    return parts


def _strip_identifier(value: str) -> str:
    clean = value.strip().replace("`", "")
    if "." in clean:
        clean = clean.split(".")[-1]
    return clean


def _normalize_for_conditions(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.replace("`", "")).lower()


def _has_bound_condition(sql: str, field: str, param_names: tuple[str, ...]) -> bool:
    escaped = re.escape(field.lower())
    placeholders = [
        rf"%\({re.escape(name.lower())}\)s"
        for name in param_names
    ]
    placeholders.extend(rf":{re.escape(name.lower())}" for name in param_names)
    placeholders.append(r"\?")
    placeholder_re = "|".join(placeholders)
    return bool(re.search(rf"\b{escaped}\s*=\s*(?:{placeholder_re})", sql))


def _tenant_field_for_table(mapping: TableMapping, table: str) -> str | None:
    if table == mapping.job_table:
        return mapping.job_fields.get("tenant_id")
    if table == mapping.attendance_table:
        return mapping.attendance_fields.get("tenant_id")
    return None
