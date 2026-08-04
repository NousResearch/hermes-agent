"""Postiz DB bridge: insert approved drafts into Postiz PostgreSQL.

If Postiz social integrations are not linked, falls back to manual export
so approved drafts are still usable (printed to stdout / saved to file).
"""
import os
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

try:
    import psycopg2
except ImportError:
    psycopg2 = None

POSTGRES_HOST = os.getenv("POSTIZ_DB_HOST", "127.0.0.1")
POSTGRES_PORT = int(os.getenv("POSTIZ_DB_PORT", "5432"))
POSTGRES_USER = os.getenv("POSTIZ_DB_USER", "postiz-user")
POSTGRES_PASS = os.getenv("POSTIZ_DB_PASS", "postiz-password")
POSTGRES_DB = os.getenv("POSTIZ_DB_NAME", "postiz-db-local")

# Organisation ID from the existing Postiz instance
ORG_ID = "2645662d-a479-4a6a-91ca-a50a7d29f607"

# Integration IDs per brand/platform — restored from Postiz DB backup 03/07/26.
# All 4 X accounts have long-lived OAuth1 tokens (expire 2058).
# LinkedIn has OAuth2 with refresh token (Postiz auto-refreshes).
# Focus: personal accounts only (sahil_twitter, sahil_linkedin).
# App brand accounts (matchdaymaestro, plenishd, coachos) are wired but paused.
INTEGRATION_MAP: dict[str, Optional[str]] = {
    # Personal accounts (ACTIVE)
    "sahil_twitter_twitter": "cmp8jnrcs0003oa6vxtjfs4et",
    "sahil_linkedin_linkedin": "cmp8v51dh0001nz6y5nmydws4",
    # App brand accounts (PAUSED — not in use yet)
    "matchdaymaestro_twitter": "cmp8i1isr0001oa6v7s7luvt8",
    "plenishd_twitter": "cmp8kogmw0005oa6vrhzr240m",
    "coachos_twitter": "cmp8nxc5y0007oa6vkx8qmvb2",
    "matchdaymaestro_instagram": "cmp94dmqb0001qo6plq4e3wf0",
    # Telegram (test channel)
    "sahil_telegram": "72a8d345-6951-4707-b503-03070d7643e3",
}


def _conn():
    if psycopg2 is None:
        raise RuntimeError("psycopg2 not installed. Install with: pip install psycopg2-binary")
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASS,
        database=POSTGRES_DB,
    )


def check_publisher_readiness() -> None:
    """Read-only proof that Postiz can accept personal-account posts."""
    conn = _conn()
    try:
        conn.set_session(readonly=True, autocommit=True)
        cur = conn.cursor()
        required = [
            INTEGRATION_MAP["sahil_twitter_twitter"],
            INTEGRATION_MAP["sahil_linkedin_linkedin"],
        ]
        cur.execute(
            """SELECT to_regclass('public.\"Post\"'), COUNT(*)
               FROM \"Integration\"
               WHERE id = ANY(%s) AND \"deletedAt\" IS NULL AND disabled = false""",
            (required,),
        )
        post_table, integration_count = cur.fetchone()
        cur.close()
        if post_table is None:
            raise RuntimeError("Postiz Post table is missing")
        if integration_count < len(required):
            raise RuntimeError("required personal Postiz integrations are unavailable")
    finally:
        conn.close()


def refresh_integration_map() -> dict[str, Optional[str]]:
    """Query the Postiz DB and rebuild INTEGRATION_MAP from live data."""
    try:
        conn = _conn()
        cur = conn.cursor()
        cur.execute(
            """SELECT id, "providerIdentifier", name
               FROM "Integration"
               WHERE "organizationId" = %s AND "deletedAt" IS NULL AND disabled = false""",
            (ORG_ID,),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        # Build provider→integrations mapping
        provider_map: dict[str, list[tuple[str, str]]] = {}
        for iid, provider, name in rows:
            provider_map.setdefault(provider.lower(), []).append((iid, name))

        print(f"Found {len(rows)} active integration(s) in Postiz DB:")
        for iid, provider, name in rows:
            print(f"  [{provider}] {name}  ({iid})")

        # Update the global INTEGRATION_MAP from DB rows
        # Map known Postiz provider identifiers to our brand+platform keys
        # This is best-effort — we match known providers to our expected keys
        updated = {}
        for key in INTEGRATION_MAP:
            # Parse key format: brand_provider
            # e.g. "matchdaymaestro_twitter" -> provider "twitter"
            parts = key.split("_", 1)
            if len(parts) == 2:
                brand, provider = parts
                # Handle duplicate brand in key (legacy format)
                if provider.startswith(brand + "_"):
                    provider = provider[len(brand) + 1:]
            else:
                continue

            matches = provider_map.get(provider, [])
            if matches:
                updated[key] = matches[0][0]  # use first match
            else:
                updated[key] = None

        INTEGRATION_MAP.clear()
        INTEGRATION_MAP.update(updated)
        return INTEGRATION_MAP

    except Exception as e:
        print(f"Could not query Postiz DB: {e}")
        print("INTEGRATION_MAP unchanged. Only Telegram is wired.")
        return INTEGRATION_MAP


def _get_integration_id(brand: str, platform: str) -> Optional[str]:
    key = f"{brand}_{platform}"
    integration_id = INTEGRATION_MAP.get(key)
    if integration_id:
        return integration_id

    # Lazy refresh: if the key exists but is None, try querying the live DB
    # This catches cases where integrations were added after the map was built.
    # Only refresh-once per call — no recursion.
    if key in INTEGRATION_MAP:
        try:
            refresh_integration_map()
            return INTEGRATION_MAP.get(key)
        except Exception:
            pass

    return None


def _manual_export(
    body_text: str,
    brand: str,
    platform: str,
    title: Optional[str] = None,
    media_path: Optional[str] = None,
) -> None:
    """Print an approved draft to stdout for manual posting.

    Written to workspace file so downstream tools / humans can find it.
    """
    export_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "data", "publish_queue"
    )
    os.makedirs(export_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_title = (title or "untitled").replace(" ", "_").replace("/", "-")[:40]
    filename = f"{timestamp}_{brand}_{platform}_{safe_title}.txt"
    filepath = os.path.join(export_dir, filename)

    content_lines = [
        f"# Publish Queue — {brand} / {platform}",
        f"# Generated: {datetime.now(timezone.utc).isoformat()}",
        f"# Title: {title or ''}",
        f"# Media: {media_path or '(none)'}",
        "",
        body_text,
        "",
        "# --- Copy the text above, attach the media file, paste into the platform ---",
    ]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(content_lines))

    print(f"  Manual export saved: {filepath}")
    print(f"  {'='*60}")
    print(f"  Brand:  {brand}")
    print(f"  Platform: {platform}")
    if title:
        print(f"  Title:  {title}")
    print()
    print(body_text)
    print()
    print(f"  {'='*60}")


def list_postiz_integrations() -> list[dict]:
    """Return active Postiz integrations as a debug list."""
    try:
        conn = _conn()
        cur = conn.cursor()
        cur.execute(
            """SELECT id, name, type, "providerIdentifier", disabled
               FROM "Integration"
               WHERE "organizationId" = %s AND "deletedAt" IS NULL""",
            (ORG_ID,),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [
            {"id": r[0], "name": r[1], "type": r[2], "provider": r[3], "disabled": r[4]}
            for r in rows
        ]
    except Exception:
        return []


def _upload_media_to_postiz(media_path: str) -> Optional[str]:
    """Upload an image to Postiz via the public API and return the media URL.
    
    Postiz stores uploads at /uploads/YYYY/MM/DD/<hash>.<ext> inside the container.
    We copy the file into the container's upload directory and return the URL path.
    """
    if not media_path or not os.path.exists(media_path):
        return None
    
    try:
        from pathlib import Path
        import hashlib
        
        now = datetime.now(timezone.utc)
        year = now.strftime("%Y")
        month = now.strftime("%m")
        day = now.strftime("%d")
        
        # Read the file and hash it for the filename
        with open(media_path, "rb") as f:
            file_data = f.read()
        file_hash = hashlib.md5(file_data).hexdigest()
        ext = Path(media_path).suffix.lower()
        
        rel_path = f"{year}/{month}/{day}/{file_hash}{ext}"
        container_path = f"/uploads/{rel_path}"
        
        # Copy into the container
        import subprocess
        result = subprocess.run(
            ["docker", "cp", media_path, f"postiz:{container_path}"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            # Create the directory first
            subprocess.run(
                ["docker", "exec", "postiz", "mkdir", "-p", f"/uploads/{year}/{month}/{day}"],
                capture_output=True, text=True, timeout=10,
            )
            result = subprocess.run(
                ["docker", "cp", media_path, f"postiz:{container_path}"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                print(f"  Media upload failed: {result.stderr[:100]}")
                return None
        
        # Postiz serves uploads at /uploads/<path> via nginx
        return f"/uploads/{rel_path}"
    except Exception as e:
        print(f"  Media upload error: {e}")
        return None


def queue_post(
    body_text: str,
    brand: str,
    platform: str,
    title: Optional[str] = None,
    publish_at: Optional[datetime] = None,
    group: str = "kensei-generated",
    state: str = "QUEUE",
    media_path: Optional[str] = None,
) -> Optional[str]:
    """Insert a post into Postiz DB, or export manually if no integration.

    ``media_path`` is the local image/video to attach. If provided, the file is
    copied into the Postiz container's /uploads directory and the URL is stored
    in the Post.image column.

    Returns the Postiz post ID if queued, or None (manual export printed).
    """
    integration_id = _get_integration_id(brand, platform)
    if not integration_id:
        _manual_export(body_text, brand, platform, title=title, media_path=media_path)
        return None

    # Upload media if provided
    image_url = None
    if media_path:
        image_url = _upload_media_to_postiz(media_path)

    post_id = str(uuid.uuid4())
    scheduled = publish_at or (datetime.now(timezone.utc) + timedelta(hours=2))

    conn = _conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO "Post" (
                id, state, "publishDate", "organizationId", "integrationId",
                content, title, "group", delay, "approvedSubmitForOrder",
                "createdAt", "updatedAt", image
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, 0, 'NO',
                NOW(), NOW(), %s
            )
            """,
            (post_id, state, scheduled, ORG_ID, integration_id, body_text, title or "", group, image_url),
        )
        conn.commit()
        cur.close()
        print(f"  Queued in Postiz: {post_id} ({brand}/{platform})")
        return post_id
    except Exception as e:
        print(f"  Postiz insert failed: {e}")
        print(f"  Falling back to manual export.")
        _manual_export(body_text, brand, platform, title=title, media_path=media_path)
        return None
    finally:
        conn.close()


def list_unpublished(brand: Optional[str] = None, limit: int = 50) -> list:
    conn = _conn()
    cur = conn.cursor()
    integration_ids = [v for v in INTEGRATION_MAP.values() if v]
    if not integration_ids:
        cur.execute(
            """
            SELECT id, content, title, "integrationId", state, "publishDate"
            FROM "Post"
            WHERE state = 'QUEUE'
            ORDER BY "publishDate" ASC
            LIMIT %s
            """,
            (limit,),
        )
    else:
        placeholders = ",".join(["%s"] * len(integration_ids))
        q = f"""
            SELECT id, content, title, "integrationId", state, "publishDate"
            FROM "Post"
            WHERE "integrationId" IN ({placeholders}) AND state = 'QUEUE'
            ORDER BY "publishDate" ASC
            LIMIT %s
        """
        cur.execute(q, integration_ids + [limit])
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(zip(["id", "content", "title", "integration_id", "state", "publish_date"], r)) for r in rows]


if __name__ == "__main__":
    import sys
    if "--refresh-integrations" in sys.argv:
        print("Refreshing INTEGRATION_MAP from Postiz DB...")
        result = refresh_integration_map()
        print(f"\nINTEGRATION_MAP now has {sum(1 for v in result.values() if v)} live integration(s).")
        for k, v in result.items():
            status = f"✓ {v[:12]}..." if v else "✗ not linked"
            print(f"  {k:35s} {status}")
    elif "--list-integrations" in sys.argv:
        integrations = list_postiz_integrations()
        if integrations:
            print(f"Active Postiz integrations ({len(integrations)}):")
            for i in integrations:
                print(f"  [{i['provider']:15s}] {i['name']:30s} {i['id'][:12]}...")
        else:
            print("No active Postiz integrations found.")
            print("Social accounts must be linked via the Postiz web UI (port 8080).")
    else:
        print("postiz_bridge.py — Postiz DB bridge for KENSEI Content Engine")
        print()
        print("Usage:")
        print("  --refresh-integrations   Rebuild INTEGRATION_MAP from live DB")
        print("  --list-integrations      List active Postiz social integrations")
        print()
        print(f"Currently wired integrations: {sum(1 for v in INTEGRATION_MAP.values() if v)} / {len(INTEGRATION_MAP)}")
        print("Telegram: ✓ (Test Telegram Channel)")
        print("All other platforms: ✗ (must be linked via Postiz web UI)")
