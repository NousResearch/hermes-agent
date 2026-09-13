"""Execute PostgreSQL trigger SQL bodies on SQLite, adapting only the DDL wrapper.

This deliberately does not emulate the projection algorithm: predicates and
row mutations come from the production statements passed by initialization.
Native PostgreSQL tests separately cover its procedural parser and engine.
"""

import re


class TriggerDDL:
    def __init__(self, db):
        self.db = db
        self.functions = {}

    def execute(self, sql):
        function = re.fullmatch(
            r"\s*CREATE OR REPLACE FUNCTION (\w+)\(\) RETURNS trigger\s+"
            r"LANGUAGE plpgsql AS \$hermes\$\s+BEGIN\s+(.*?)"
            r"\s+RETURN NULL;\s+END;\s+\$hermes\$\s*", sql, re.S | re.I,
        )
        if function:
            self.functions[function[1]] = function[2]
            return self.db.execute("SELECT NULL WHERE 0")
        trigger = re.fullmatch(
            r"\s*CREATE OR REPLACE TRIGGER (\w+)\s+AFTER (.*?) ON (\w+)\s+"
            r"FOR EACH ROW WHEN \((.*?)\)\s+EXECUTE FUNCTION (\w+)\(\)\s*",
            sql, re.S | re.I,
        )
        if trigger:
            name, event, table, condition, function_name = trigger.groups()
            body = self.functions[function_name]
            self.db.execute(f"DROP TRIGGER IF EXISTS {name}")
            return self.db.execute(
                f"CREATE TRIGGER {name} AFTER {event} ON {table} "
                f"FOR EACH ROW WHEN ({condition}) BEGIN {body} END"
            )
        return None
