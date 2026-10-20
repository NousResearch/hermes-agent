#!/usr/bin/env python3
"""Database schema introspection for Hermes Agent."""
import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any

# Reuse sql_exec drivers
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sql-query-executor" / "scripts"))
from sql_exec import get_driver, BaseDriver


SCHEMA_QUERIES = {
    "postgresql": {
        "tables": """
            SELECT 
                t.table_name,
                t.table_type,
                obj_description(c.oid) as table_comment
            FROM information_schema.tables t
            LEFT JOIN pg_class c ON c.relname = t.table_name
            WHERE t.table_schema = 'public'
            ORDER BY t.table_name
        """,
        "columns": """
            SELECT 
                c.table_name,
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.column_default,
                c.ordinal_position,
                pgd.description as column_comment
            FROM information_schema.columns c
            LEFT JOIN pg_catalog.pg_stato_all_tables st ON st.relname = c.table_name
            LEFT JOIN pg_catalog.pg_description pgd ON pgd.objoid = st.relid AND pgd.objsubid = c.ordinal_position
            WHERE c.table_schema = 'public'
            ORDER BY c.table_name, c.ordinal_position
        """,
        "foreign_keys": """
            SELECT
                tc.table_name,
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name,
                tc.constraint_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage ccu ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public'
        """,
        "indexes": """
            SELECT
                schemaname,
                tablename,
                indexname,
                indexdef
            FROM pg_indexes
            WHERE schemaname = 'public'
        """
    },
    "mysql": {
        "tables": """
            SELECT 
                TABLE_NAME as table_name,
                TABLE_TYPE as table_type,
                TABLE_COMMENT as table_comment
            FROM information_schema.TABLES
            WHERE TABLE_SCHEMA = DATABASE()
            ORDER BY TABLE_NAME
        """,
        "columns": """
            SELECT 
                TABLE_NAME as table_name,
                COLUMN_NAME as column_name,
                DATA_TYPE as data_type,
                IS_NULLABLE as is_nullable,
                COLUMN_DEFAULT as column_default,
                ORDINAL_POSITION as ordinal_position,
                COLUMN_COMMENT as column_comment
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
            ORDER BY TABLE_NAME, ORDINAL_POSITION
        """,
        "foreign_keys": """
            SELECT
                TABLE_NAME as table_name,
                COLUMN_NAME as column_name,
                REFERENCED_TABLE_NAME as foreign_table_name,
                REFERENCED_COLUMN_NAME as foreign_column_name,
                CONSTRAINT_NAME as constraint_name
            FROM information_schema.KEY_COLUMN_USAGE
            WHERE REFERENCED_TABLE_NAME IS NOT NULL AND TABLE_SCHEMA = DATABASE()
        """,
        "indexes": """
            SELECT
                TABLE_SCHEMA as schemaname,
                TABLE_NAME as tablename,
                INDEX_NAME as indexname,
                GROUP_CONCAT(COLUMN_NAME ORDER BY SEQ_IN_INDEX) as indexdef
            FROM information_schema.STATISTICS
            WHERE TABLE_SCHEMA = DATABASE()
            GROUP BY TABLE_SCHEMA, TABLE_NAME, INDEX_NAME
        """
    },
    "sqlite": {
        "tables": """
            SELECT 
                name as table_name,
                'BASE TABLE' as table_type,
                '' as table_comment
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """,
        "columns": """
            -- SQLite PRAGMA table_info per table, handled separately
            SELECT 1
        """,
        "foreign_keys": """
            -- SQLite PRAGMA foreign_key_list per table, handled separately
            SELECT 1
        """,
        "indexes": """
            SELECT 
                'main' as schemaname,
                m.name as tablename,
                i.name as indexname,
                i.sql as indexdef
            FROM sqlite_master i
            JOIN sqlite_master m ON m.name = i.tbl_name
            WHERE i.type = 'index' AND m.type = 'table'
        """
    },
    "clickhouse": {
        "tables": """
            SELECT 
                name as table_name,
                engine as table_type,
                comment as table_comment
            FROM system.tables
            WHERE database = currentDatabase()
            ORDER BY name
        """,
        "columns": """
            SELECT 
                table as table_name,
                name as column_name,
                type as data_type,
                '' as is_nullable,
                default_expression as column_default,
                position as ordinal_position,
                comment as column_comment
            FROM system.columns
            WHERE database = currentDatabase()
            ORDER BY table, position
        """,
        "foreign_keys": """
            SELECT 1
        """,
        "indexes": """
            SELECT 1
        """
    }
}


def detect_db_type(dsn: str) -> str:
    if dsn.startswith("postgresql://") or dsn.startswith("postgres://"):
        return "postgresql"
    elif dsn.startswith("mysql://"):
        return "mysql"
    elif dsn.startswith("sqlite://"):
        return "sqlite"
    elif dsn.startswith("clickhouse://"):
        return "clickhouse"
    else:
        raise ValueError(f"Unknown DSN type: {dsn}")


def get_schema_data(driver: BaseDriver, db_type: str, dsn: str) -> Dict[str, Any]:
    queries = SCHEMA_QUERIES[db_type]
    schema = {"tables": [], "columns": {}, "foreign_keys": [], "indexes": []}
    
    # Tables
    _, tables = driver.execute(queries["tables"], {}, limit=0)
    schema["tables"] = [dict(zip(["table_name", "table_type", "table_comment"], row)) for row in tables]
    
    # Columns
    if db_type == "sqlite":
        # SQLite needs per-table PRAGMA
        for table in schema["tables"]:
            tname = table["table_name"]
            _, cols = driver.execute(f"PRAGMA table_info({tname})", {}, limit=0)
            schema["columns"][tname] = [dict(zip(["cid", "column_name", "data_type", "notnull", "default_value", "pk"], row)) for row in cols]
            # Foreign keys
            _, fks = driver.execute(f"PRAGMA foreign_key_list({tname})", {}, limit=0)
            for fk in fks:
                schema["foreign_keys"].append({
                    "table_name": tname,
                    "column_name": fk[3],  # from
                    "foreign_table_name": fk[2],  # table
                    "foreign_column_name": fk[4],  # to
                    "constraint_name": fk[0]  # id
                })
    else:
        _, cols = driver.execute(queries["columns"], {}, limit=0)
        col_names = ["table_name", "column_name", "data_type", "is_nullable", "column_default", "ordinal_position", "column_comment"]
        for row in cols:
            col_dict = dict(zip(col_names, row))
            tname = col_dict["table_name"]
            if tname not in schema["columns"]:
                schema["columns"][tname] = []
            schema["columns"][tname].append(col_dict)
        
        # Foreign keys
        _, fks = driver.execute(queries["foreign_keys"], {}, limit=0)
        fk_names = ["table_name", "column_name", "foreign_table_name", "foreign_column_name", "constraint_name"]
        schema["foreign_keys"] = [dict(zip(fk_names, row)) for row in fks]
    
    # Indexes
    if db_type != "sqlite" or db_type in SCHEMA_QUERIES:
        _, idxs = driver.execute(queries["indexes"], {}, limit=0)
        if db_type == "postgresql":
            idx_names = ["schemaname", "tablename", "indexname", "indexdef"]
        elif db_type == "mysql":
            idx_names = ["schemaname", "tablename", "indexname", "indexdef"]
        elif db_type == "clickhouse":
            idx_names = []
        else:
            idx_names = ["schemaname", "tablename", "indexname", "indexdef"]
        if idx_names and idxs:
            schema["indexes"] = [dict(zip(idx_names, row)) for row in idxs]
    
    return schema


def output_json(schema: Dict, output: str = None):
    data = json.dumps(schema, ensure_ascii=False, indent=2)
    if output:
        Path(output).write_text(data)
    else:
        print(data)


def output_markdown(schema: Dict, tables_only: bool = False, table_filter: str = None, output: str = None):
    lines = []
    
    if not tables_only:
        lines.append("# Database Schema\n")
        lines.append(f"**Tables:** {len(schema['tables'])}\n")
    
    tables_to_show = schema["tables"]
    if table_filter:
        tables_to_show = [t for t in tables_to_show if t["table_name"] == table_filter]
    
    for table in tables_to_show:
        tname = table["table_name"]
        lines.append(f"## Table: `{tname}`")
        if table.get("table_comment"):
            lines.append(f"> {table['table_comment']}")
        lines.append("")
        
        # Columns
        lines.append("| Column | Type | Nullable | Default | Comment |")
        lines.append("|--------|------|----------|---------|---------|")
        for col in schema["columns"].get(tname, []):
            cname = col.get("column_name", "")
            ctype = col.get("data_type", "")
            nullable = "YES" if col.get("is_nullable", "YES") == "YES" else "NO"
            default = col.get("column_default", "") or ""
            comment = col.get("column_comment", "") or ""
            lines.append(f"| {cname} | {ctype} | {nullable} | {default} | {comment} |")
        lines.append("")
        
        # Foreign keys
        fks = [fk for fk in schema["foreign_keys"] if fk["table_name"] == tname]
        if fks:
            lines.append("### Foreign Keys")
            lines.append("| Column | References |")
            lines.append("|--------|------------|")
            for fk in fks:
                lines.append(f"| {fk['column_name']} | {fk['foreign_table_name']}.{fk['foreign_column_name']} |")
            lines.append("")
        
        # Indexes
        idxs = [i for i in schema["indexes"] if i.get("tablename") == tname or i.get("table_name") == tname]
        if idxs:
            lines.append("### Indexes")
            lines.append("| Name | Definition |")
            lines.append("|------|------------|")
            for idx in idxs:
                iname = idx.get("indexname", "")
                idef = idx.get("indexdef", "")
                lines.append(f"| {iname} | {idef} |")
            lines.append("")
    
    result = "\n".join(lines)
    if output:
        Path(output).write_text(result)
    else:
        print(result)


def output_mermaid(schema: Dict, output: str = None):
    lines = ["erDiagram"]
    
    # Entities
    for table in schema["tables"]:
        tname = table["table_name"]
        cols = schema["columns"].get(tname, [])
        lines.append(f"    {tname} {{")
        for col in cols:
            cname = col.get("column_name", "")
            ctype = col.get("data_type", "")
            pk = "PK" if col.get("pk") == 1 or col.get("ordinal_position") == 1 else ""
            lines.append(f"        {ctype} {cname} {pk}")
        lines.append("    }")
    
    # Relationships
    for fk in schema["foreign_keys"]:
        lines.append(f"    {fk['table_name']} ||--o{{ {fk['foreign_table_name']} : \"{fk['column_name']}\"")
    
    result = "\n".join(lines)
    if output:
        Path(output).write_text(result)
    else:
        print(result)


def main():
    parser = argparse.ArgumentParser(description="Database schema introspection")
    parser.add_argument("--dsn", help="Database DSN (or use DATABASE_DSN env)")
    parser.add_argument("--format", choices=["json", "markdown", "mermaid"], default="markdown")
    parser.add_argument("--tables-only", action="store_true", help="Only show tables list")
    parser.add_argument("--table", help="Filter to specific table")
    parser.add_argument("--output", help="Output file path")
    args = parser.parse_args()

    dsn = args.dsn or os.environ.get("DATABASE_DSN")
    if not dsn:
        parser.error("DSN required via --dsn or DATABASE_DSN environment variable")

    db_type = detect_db_type(dsn)
    driver = get_driver(dsn)
    schema = get_schema_data(driver, db_type, dsn)

    if args.format == "json":
        output_json(schema, args.output)
    elif args.format == "markdown":
        output_markdown(schema, args.tables_only, args.table, args.output)
    elif args.format == "mermaid":
        output_mermaid(schema, args.output)


if __name__ == "__main__":
    main()