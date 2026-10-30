"""Disposable synthetic catalog. No third-party services or authentication."""

import csv
from contextlib import contextmanager
import io
import json
import sqlite3
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


class Catalog:
    def __init__(self, database, injected_export_bug=False):
        self.database = Path(database)
        self.injected_export_bug = injected_export_bug
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE categories(run TEXT, id INTEGER, PRIMARY KEY(run,id));
                CREATE TABLE products(run TEXT, logical_id INTEGER, name TEXT, price INTEGER,
                  category INTEGER, PRIMARY KEY(run,logical_id),
                  FOREIGN KEY(run,category) REFERENCES categories(run,id));
            """)
            db.execute("INSERT INTO categories VALUES('sentinel',0)")
            db.execute("INSERT INTO products VALUES('sentinel',0,'Unrelated sentinel',123,0)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database)
        try:
            db.execute("PRAGMA foreign_keys=ON")
            with db:
                yield db
        finally:
            db.close()

    def counts(self, run):
        with self.connect() as db:
            products = db.execute("SELECT COUNT(*) FROM products WHERE run=?", (run,)).fetchone()[0]
            categories = db.execute("SELECT COUNT(*) FROM categories WHERE run=?", (run,)).fetchone()[0]
            orphaned = db.execute("SELECT COUNT(*) FROM products p LEFT JOIN categories c ON p.run=c.run AND p.category=c.id WHERE p.run=? AND c.id IS NULL", (run,)).fetchone()[0]
            sentinel = db.execute("SELECT name,price,category FROM products WHERE run='sentinel' AND logical_id=0").fetchone()
        return {"products": products, "categories": categories, "orphaned": orphaned, "sentinel_intact": sentinel == ("Unrelated sentinel", 123, 0)}

    def prepare(self, run, rows):
        if not run.startswith("volume-") or len(rows) > 250:
            raise ValueError("Fixture accepts run-owned batches of at most 250")
        with self.connect() as db:
            db.executemany("INSERT OR IGNORE INTO categories VALUES(?,?)", [(run, i) for i in range(8)])
            for row in rows:
                values = (row["id"], row["name"], row["price"], row["category"])
                existing = db.execute("SELECT logical_id,name,price,category FROM products WHERE run=? AND logical_id=?", (run, row["id"])).fetchone()
                if existing is not None and existing != values:
                    raise ValueError("Idempotency conflict: same identity, different content")
                db.execute("INSERT OR IGNORE INTO products VALUES(?,?,?,?,?)", (run, *values))
        return self.counts(run)

    def cleanup(self, run):
        if not run.startswith("volume-"):
            raise ValueError("Refusing cleanup outside fixture run namespace")
        with self.connect() as db:
            db.execute("DELETE FROM products WHERE run=?", (run,))
            db.execute("DELETE FROM categories WHERE run=?", (run,))
        return self.counts(run)

    def query(self, run, q="", category="", sort="name", page=1, page_size=50, export=False):
        if sort not in ("name", "price") or page < 1 or not 1 <= page_size <= 200:
            raise ValueError("Invalid pagination/sort")
        where, params = "run=? AND instr(name,?)>0", [run, q]
        if category != "":
            where += " AND category=?"
            params.append(int(category))
        with self.connect() as db:
            total = db.execute(f"SELECT COUNT(*) FROM products WHERE {where}", params).fetchone()[0]
            sql = f"SELECT logical_id,name,price,category FROM products WHERE {where} ORDER BY {sort},logical_id"
            if not export:
                sql += " LIMIT ? OFFSET ?"
                params += [page_size, (page - 1) * page_size]
            elif self.injected_export_bug and total >= 1000:
                # Deliberately injected fixture-only defect: a full export is truncated.
                sql += " LIMIT ?"
                params += [page_size]
            rows = [dict(zip(("id", "name", "price", "category"), row)) for row in db.execute(sql, params)]
        return {"total": total, "rows": rows}


def csv_text(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=("id", "name", "price", "category"))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def server_for(catalog):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def do_GET(self):
            parsed = urlparse(self.path)
            args = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            if parsed.path == "/favicon.ico":
                return self.respond(204, "", "image/x-icon")
            if parsed.path == "/":
                return self.respond(200, (Path(__file__).with_name("index.html")).read_text(encoding="utf-8"), "text/html")
            try:
                run = args.pop("run")
                if parsed.path == "/counts":
                    return self.respond(200, catalog.counts(run))
                result = catalog.query(run, page=int(args.pop("page", 1)), page_size=int(args.pop("page_size", 50)), export=parsed.path == "/export", **args)
                if parsed.path == "/export":
                    return self.respond(200, csv_text(result["rows"]), "text/csv")
                if parsed.path != "/products":
                    return self.respond(404, {"error": "Unknown endpoint"})
                self.respond(200, result)
            except (KeyError, TypeError, ValueError) as error:
                self.respond(400, {"error": str(error)})

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 100000:
                    raise ValueError("Bounded JSON body required")
                data = json.loads(self.rfile.read(size))
                if self.path == "/prepare":
                    result = catalog.prepare(data["run"], data["rows"])
                elif self.path == "/cleanup":
                    result = catalog.cleanup(data["run"])
                else:
                    return self.respond(404, {"error": "Unknown endpoint"})
                self.respond(200, result)
            except (KeyError, TypeError, ValueError, sqlite3.Error) as error:
                self.respond(400, {"error": str(error)})

        def respond(self, status, value, content_type="application/json"):
            payload = json.dumps(value, ensure_ascii=False) if content_type == "application/json" else value
            raw = payload.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    return HTTPServer(("127.0.0.1", 0), Handler)
