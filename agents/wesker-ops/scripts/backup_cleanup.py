#!/usr/bin/env python3
import os
import shutil
import time
import logging

# Configuration
BACKUP_DIR = "/home/kensei/backups/"
EXCLUSIONS = {"daily", "sessions"}
RETENTION_DAYS = 60
DRY_RUN = os.environ.get("DRY_RUN", "true").lower() == "true"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def cleanup():
    now = time.time()
    cutoff = now - (RETENTION_DAYS * 86400)
    
    if not os.path.exists(BACKUP_DIR):
        logging.error(f"Backup directory {BACKUP_DIR} does not exist.")
        return

    logging.info(f"Scanning {BACKUP_DIR} for items older than {RETENTION_DAYS} days...")
    if DRY_RUN:
        logging.info("DRY_RUN is enabled. No files will be deleted.")

    try:
        # List all items in the backup directory
        items = os.listdir(BACKUP_DIR)
    except OSError as e:
        logging.error(f"Failed to list directory {BACKUP_DIR}: {e}")
        return

    deleted_count = 0
    total_freed_size = 0

    for item in items:
        # Skip excluded directories
        if item in EXCLUSIONS:
            continue

        item_path = os.path.join(BACKUP_DIR, item)
        
        try:
            # Use mtime of the directory/file itself as requested
            mtime = os.path.getmtime(item_path)
            
            if mtime < cutoff:
                # Calculate size before deletion
                size = 0
                if os.path.isdir(item_path):
                    for root, dirs, files in os.walk(item_path):
                        for f in files:
                            fp = os.path.join(root, f)
                            if os.path.exists(fp):
                                size += os.path.getsize(fp)
                else:
                    size = os.path.getsize(item_path)

                if DRY_RUN:
                    logging.info(f"[DRY-RUN] Would remove: {item} (mtime: {time.ctime(mtime)}, size: {size} bytes)")
                else:
                    logging.info(f"Removing: {item} (mtime: {time.ctime(mtime)}, size: {size} bytes)")
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                
                deleted_count += 1
                total_freed_size += size
        except OSError as e:
            logging.error(f"Failed to process {item_path}: {e}")

    logging.info(f"Cleanup finished. Items processed: {deleted_count}. Total size: {total_freed_size} bytes.")

if __name__ == "__main__":
    cleanup()
