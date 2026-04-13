#!/usr/bin/env python3
import os
import yaml
import hashlib
import logging
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)

class FileResolver:
    """
    Implements a GBrain-style Cloud Storage Breadcrumbing pipeline.
    Lifecycle:
        mirror -> redirect -> resolve
    Moves heavy binary assets to S3 and replaces them with tiny YAML .redirect traces.
    """
    def __init__(self, s3_client: Any = None, bucket_name: str = None):
        self.s3_client = s3_client
        self.bucket_name = bucket_name or os.environ.get("HERMES_ASSET_BUCKET")

    def _compute_hash(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    def mirror(self, file_path: str, remote_path: str) -> bool:
        """Uploads the file to cloud storage without removing local copy."""
        if not self.s3_client or not self.bucket_name:
            logger.warning("No S3 client or bucket configured for mirror.")
            return False
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, remote_path)
            return True
        except Exception as e:
            logger.error("Failed to mirror %s: %s", file_path, e)
            return False

    def redirect(self, file_path: str, remote_path: str) -> bool:
        """Replaces local file with a .redirect YAML breadcrumb."""
        p = Path(file_path)
        if not p.exists():
            return False
            
        hash_val = self._compute_hash(p)
        mime_ext = p.suffix
        
        redir_path = p.with_suffix(".redirect")
        
        breadcrumb = {
            "version": 1,
            "target": f"s3://{self.bucket_name}/{remote_path}",
            "hash": hash_val,
            "original_name": p.name,
            "mime_type": mime_ext
        }
        
        with open(redir_path, "w") as f:
            yaml.dump(breadcrumb, f)
            
        # Clean: Remove original
        p.unlink()
        return True

    def resolve(self, requested_path: str) -> Optional[str]:
        """
        Returns the local path if it exists.
        If replaced by a breadcrumb, returns a presigned cloud URL or remote S3 path.
        """
        p = Path(requested_path)
        if p.exists():
            return str(p)
            
        redir_path = p.with_suffix(".redirect")
        if redir_path.exists():
            with open(redir_path, "r") as f:
                data = yaml.safe_load(f)
            
            target = data.get("target")
            if target and target.startswith("s3://"):
                if self.s3_client:
                    try:
                        key = target.split("/", 3)[-1]
                        url = self.s3_client.generate_presigned_url(
                            'get_object',
                            Params={'Bucket': self.bucket_name, 'Key': key},
                            ExpiresIn=3600
                        )
                        return url
                    except Exception as e:
                        logger.error("Failed to generate presigned URL: %s", e)
                return target
        return None
