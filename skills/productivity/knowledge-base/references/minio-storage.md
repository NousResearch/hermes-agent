# MinIO Source File Storage

Knowledge base source files (ePubs, PDFs, video transcripts) are stored permanently on MinIO
so Obsidian notes can reference them via stable URLs.

## Architecture

```
Obsidian Vault (Git-synced)          MinIO (local, Tailscale-accessible)
├── Knowledge base/                    └── knowledge-base/
│   ├── sante/note.md                       ├── books/
│   └── books/note.md                      │   └── parasites-humbert.epub
│       └── frontmatter:                    ├── reels/
│           source_file: https://...        │   └── DR-rVE1ggZH.mp4
│                                           └── articles/
│                                               └── threads-adhd-2026.txt
```

## Upload command (from researcher worker)

```bash
# Upload a file to MinIO
mc cp /tmp/book_parasites_full.txt \
  minio/knowledge-base/books/parasites-humbert.txt

# Get public URL (Tailscale IP)
echo "https://<minio-host>:9000/knowledge-base/books/parasites-humbert.txt"
```

## Frontmatter convention

```yaml
source_file: https://vmi3304846:9000/knowledge-base/books/parasites-humbert.epub
```

The `source_file` field is a permanent link, unlike `/tmp/` paths which may be cleaned up.
It goes at the top of the note, alongside `source_url`.

## When to upload

- **Books**: after extracting text, upload the original ePub/PDF
- **Reels**: after downloading video, upload the .mp4
- **Articles/Threads**: after extracting, upload the raw text/markdown

Always upload BEFORE writing the note so the `source_file` URL is valid from the start.
