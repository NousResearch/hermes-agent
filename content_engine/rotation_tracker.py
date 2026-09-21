"""Rotation Tracker — Visual Content Pipeline 2.0 diversity enforcement.

Tracks published image metadata and enforces soft no-repeat rules to prevent
visual fatigue across the content pipeline.

Usage:
    from rotation_tracker import RotationTracker

    tracker = RotationTracker()
    tracker.record(article_id="abc123", brand="sahil_twitter", studio="saga-noir-studio", ...)
    recent = tracker.recent_for_brand("sahil_twitter", limit=10)
    can_use = tracker.can_use(studio="saga-noir-studio", brand="sahil_twitter")
"""

from __future__ import annotations
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Optional

# Default DB path — same directory as the content engine DB
DEFAULT_DB_PATH = Path(__file__).resolve().parent / "db" / "rotation_tracker.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS image_publications (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id      TEXT NOT NULL,
    brand           TEXT NOT NULL,
    target_surface  TEXT NOT NULL DEFAULT 'social',
    studio          TEXT NOT NULL,
    intent          TEXT NOT NULL DEFAULT '',
    narrative       TEXT NOT NULL DEFAULT '',
    layout          TEXT NOT NULL DEFAULT '',
    composition     TEXT NOT NULL DEFAULT '',
    palette         TEXT NOT NULL DEFAULT '',
    baoyu_type      TEXT NOT NULL DEFAULT '',
    baoyu_style     TEXT NOT NULL DEFAULT '',
    model_used      TEXT NOT NULL DEFAULT '',
    generation_cost REAL DEFAULT 0.0,
    ocr_passed      INTEGER DEFAULT 1,
    published_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rt_brand ON image_publications(brand);
CREATE INDEX IF NOT EXISTS idx_rt_published ON image_publications(published_at);
CREATE INDEX IF NOT EXISTS idx_rt_studio ON image_publications(studio);
CREATE INDEX IF NOT EXISTS idx_rt_brand_studio ON image_publications(brand, studio);
"""


class RotationTracker:
    """Tracks published image metadata and enforces diversity rules."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.executescript(SCHEMA)
        conn.commit()
        conn.close()

    def record(self, **kwargs) -> int:
        """Record a published image's metadata.

        Required kwargs: article_id, brand, studio
        Optional kwargs: target_surface, intent, narrative, layout, composition,
                         palette, baoyu_type, baoyu_style, model_used,
                         generation_cost, ocr_passed
        """
        fields = {
            "article_id": kwargs.get("article_id", ""),
            "brand": kwargs.get("brand", ""),
            "target_surface": kwargs.get("target_surface", "social"),
            "studio": kwargs.get("studio", ""),
            "intent": kwargs.get("intent", ""),
            "narrative": kwargs.get("narrative", ""),
            "layout": kwargs.get("layout", ""),
            "composition": kwargs.get("composition", ""),
            "palette": kwargs.get("palette", ""),
            "baoyu_type": kwargs.get("baoyu_type", ""),
            "baoyu_style": kwargs.get("baoyu_style", ""),
            "model_used": kwargs.get("model_used", ""),
            "generation_cost": kwargs.get("generation_cost", 0.0),
            "ocr_passed": kwargs.get("ocr_passed", 1),
            "published_at": datetime.now(UTC).isoformat(),
        }
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """INSERT INTO image_publications
               (article_id, brand, target_surface, studio, intent, narrative,
                layout, composition, palette, baoyu_type, baoyu_style,
                model_used, generation_cost, ocr_passed, published_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            tuple(fields.values()),
        )
        conn.commit()
        row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()
        return row_id

    def recent_for_brand(self, brand: str, limit: int = 10) -> list[dict]:
        """Return the most recent N image publications for a brand."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM image_publications WHERE brand = ? ORDER BY published_at DESC LIMIT ?",
            (brand, limit),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def recent_studios(self, brand: str, limit: int = 3) -> list[str]:
        """Return the most recent N studio names for a brand (for rotation)."""
        rows = self.recent_for_brand(brand, limit)
        return [r["studio"] for r in rows]

    def recent_layouts(self, brand: str, limit: int = 5) -> list[str]:
        """Return the most recent N layout names for a brand."""
        rows = self.recent_for_brand(brand, limit)
        return [r["layout"] for r in rows if r.get("layout")]

    def recent_compositions(self, brand: str, limit: int = 5) -> list[str]:
        """Return the most recent N composition names for a brand."""
        rows = self.recent_for_brand(brand, limit)
        return [r["composition"] for r in rows if r.get("composition")]

    def recent_palettes(self, brand: str, limit: int = 5) -> list[str]:
        """Return the most recent N palette names for a brand."""
        rows = self.recent_for_brand(brand, limit)
        return [r["palette"] for r in rows if r.get("palette")]

    def recent_history(self, brand: str, limit: int = 10) -> list[tuple]:
        """Return recent (studio, layout, composition) tuples for rotation.

        This is the format expected by image_router.decide().
        """
        rows = self.recent_for_brand(brand, limit)
        return [
            (r.get("studio", ""), r.get("layout", ""), r.get("composition", ""))
            for r in rows
        ]

    def can_use(self, *, studio: str = "", layout: str = "",
                composition: str = "", palette: str = "",
                brand: str = "") -> dict:
        """Check if a combination violates rotation rules.

        Returns a dict with 'allowed' (bool) and 'reasons' (list of strings).

        Rules:
        - Same studio within last 3 images for same brand: warn
        - Same layout within last 5 images: warn
        - Same studio + layout pair within last 10 images: warn
        - Same palette within last 5 images: warn
        """
        if not brand:
            return {"allowed": True, "reasons": []}

        reasons = []
        recent = self.recent_for_brand(brand, limit=10)

        if studio:
            recent_studios = [r["studio"] for r in recent[:3]]
            if studio in recent_studios:
                reasons.append(f"Studio '{studio}' used in last 3 images for brand '{brand}'")

        if layout:
            recent_layouts = [r["layout"] for r in recent[:5]]
            if layout in recent_layouts:
                reasons.append(f"Layout '{layout}' used in last 5 images for brand '{brand}'")

        if studio and layout:
            recent_pairs = [(r["studio"], r["layout"]) for r in recent[:10]]
            if (studio, layout) in recent_pairs:
                reasons.append(f"Studio+layout pair '{studio}+{layout}' used in last 10 images")

        if palette:
            recent_palettes = [r["palette"] for r in recent[:5]]
            if palette in recent_palettes:
                reasons.append(f"Palette '{palette}' used in last 5 images for brand '{brand}'")

        return {
            "allowed": len(reasons) == 0,
            "reasons": reasons,
        }

    def count(self, brand: Optional[str] = None) -> int:
        """Total number of tracked publications, optionally filtered by brand."""
        conn = sqlite3.connect(str(self.db_path))
        if brand:
            row = conn.execute(
                "SELECT COUNT(*) FROM image_publications WHERE brand = ?", (brand,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM image_publications").fetchone()
        conn.close()
        return row[0] if row else 0

    def prune(self, retention_days: int = 90) -> int:
        """Delete records older than retention_days. Returns count deleted."""
        conn = sqlite3.connect(str(self.db_path))
        cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
        cur = conn.execute(
            "DELETE FROM image_publications WHERE published_at < ?",
            (cutoff,),
        )
        deleted = cur.rowcount
        conn.commit()
        conn.close()
        return deleted
