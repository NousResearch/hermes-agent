"""
Postiz publisher — inserts posts directly into Postiz database.
The POST /api/posts endpoint is broken (rejects valid payloads),
so we write directly to the DB.

DEPRECATED (C026/C027): This module is a duplicate enqueue route.
The single authoritative Postiz enqueue route is
``publish_to_postiz.py`` via ``postiz_bridge.queue_post``. Approval is
state-only; the publish_to_postiz cron polls approved drafts and enqueues
them with idempotent claim guards. This class is kept for read-only
queries (get_upcoming_posts, get_post_counts_by_state, delete_post) but
its insert_post / insert_posts_batch methods should NOT be called for
enqueue — use postiz_bridge.queue_post instead.
"""

import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor


class PostizPublisher:
    """Direct DB publisher for Postiz content scheduling."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        dbname: str = "postiz-db-local",
        user: str = "postiz-user",
        password: str = os.getenv("POSTIZ_DB_PASS", "postiz-password"),
    ):
        # Docker container exposes postgres internally; default credentials from compose
        self.conn_params = {
            "host": host,
            "port": port,
            "dbname": dbname,
            "user": user,
            "password": password,
        }
    
    def _connect(self):
        """Establish database connection."""
        return psycopg2.connect(**self.conn_params, cursor_factory=RealDictCursor)
    
    def get_integrations(self, org_id: str) -> list[dict]:
        """Get all social media integrations for an organisation."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, "providerIdentifier", disabled, token
                    FROM "Integration"
                    WHERE "organizationId" = %s AND type = 'social'
                    ORDER BY name;
                    """,
                    (org_id,),
                )
                return cur.fetchall()
    
    def insert_post(
        self,
        content: str,
        org_id: str,
        integration_id: str,
        publish_at: datetime,
        title: str = "",
        image_path: str | None = None,
        video_path: str | None = None,
        state: str = "QUEUE",
        group: str | None = None,
    ) -> str:
        """
        Insert a post directly into Postiz database.
        
        Args:
            content: Post text content
            org_id: Organisation ID
            integration_id: Social media integration ID
            publish_at: Scheduled publish date/time
            title: Optional post title
            image_path: Optional path to image asset
            video_path: Optional path to video asset
            state: Post state (QUEUE, DRAFT, SCHEDULED)
            group: Content group name for batching
        
        Returns:
            Post ID
        """
        post_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        
        # Handle image/video paths
        image_url = None
        if image_path and os.path.exists(image_path):
            # Postiz stores image as relative URL or path
            image_url = image_path
        
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO "Post" (
                        id, state, "publishDate", "organizationId", "integrationId",
                        content, title, "group", delay, "createdAt", "updatedAt",
                        image
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (
                        post_id,
                        state,
                        publish_at,
                        org_id,
                        integration_id,
                        content,
                        title,
                        group or f"batch_{now.strftime('%Y%m%d')}",
                        0,
                        now,
                        now,
                        image_url,
                    ),
                )
                result = cur.fetchone()
                conn.commit()
                return result["id"]
    
    def insert_posts_batch(
        self,
        posts: list[dict],
        org_id: str,
    ) -> list[str]:
        """
        Insert multiple posts in a single transaction.
        
        Each post dict should have:
            - content (str, required)
            - integration_id (str, required)
            - publish_at (datetime, required)
            - title (str, optional)
            - image_path (str, optional)
            - video_path (str, optional)
            - state (str, optional, default QUEUE)
            - group (str, optional)
        
        Returns:
            List of inserted post IDs
        """
        post_ids = []
        now = datetime.now(UTC)
        
        with self._connect() as conn:
            with conn.cursor() as cur:
                for post in posts:
                    post_id = str(uuid.uuid4())
                    publish_at = post["publish_at"]
                    if isinstance(publish_at, str):
                        publish_at = datetime.fromisoformat(publish_at)
                    
                    image_url = None
                    if post.get("image_path") and os.path.exists(post["image_path"]):
                        image_url = post["image_path"]
                    
                    cur.execute(
                        """
                        INSERT INTO "Post" (
                            id, state, "publishDate", "organizationId", "integrationId",
                            content, title, "group", delay, "createdAt", "updatedAt",
                            image
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id;
                        """,
                        (
                            post_id,
                            post.get("state", "QUEUE"),
                            publish_at,
                            org_id,
                            post["integration_id"],
                            post["content"],
                            post.get("title", ""),
                            post.get("group", f"batch_{now.strftime('%Y%m%d')}"),
                            0,
                            now,
                            now,
                            image_url,
                        ),
                    )
                    result = cur.fetchone()
                    post_ids.append(result["id"])
                
                conn.commit()
        
        return post_ids
    
    def get_upcoming_posts(self, org_id: str, days: int = 7) -> list[dict]:
        """Get posts scheduled for the next N days."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, state, "publishDate", content, title, image, "group"
                    FROM "Post"
                    WHERE "organizationId" = %s
                      AND "publishDate" >= NOW()
                      AND "publishDate" <= NOW() + INTERVAL '%s days'
                      AND deletedAt IS NULL
                    ORDER BY "publishDate" ASC;
                    """,
                    (org_id, days),
                )
                return cur.fetchall()
    
    def delete_post(self, post_id: str) -> bool:
        """Soft-delete a post."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE "Post"
                    SET deletedAt = NOW(), state = 'DELETED'
                    WHERE id = %s;
                    """,
                    (post_id,),
                )
                conn.commit()
                return cur.rowcount > 0
    
    def get_post_counts_by_state(self, org_id: str) -> dict:
        """Get count of posts by state."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT state, COUNT(*) as count
                    FROM "Post"
                    WHERE "organizationId" = %s AND "deletedAt" IS NULL
                    GROUP BY state;
                    """,
                    (org_id,),
                )
                return {row["state"]: row["count"] for row in cur.fetchall()}


def get_default_publisher() -> PostizPublisher:
    """Factory for default publisher using Docker-exposed postgres."""
    return PostizPublisher(
        host="localhost",
        port=5432,
        dbname=os.getenv("POSTIZ_DB_NAME", "postiz-db-local"),
        user=os.getenv("POSTIZ_DB_USER", "postiz-user"),
        password=os.getenv("POSTIZ_DB_PASS", "postiz-password"),
    )
