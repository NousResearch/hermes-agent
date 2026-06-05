# MinIO Upload for Knowledge Base

Upload source files (ePub, PDF, etc.) to MinIO after extraction.

## Credentials

From researcher profile env:
- `MINIO_ENDPOINT=http://localhost:9000`
- `MINIO_ACCESS_KEY`
- `MINIO_SECRET_KEY`
- `MINIO_BUCKET=knowledge-base`

## Upload

```bash
source /root/.hermes/profiles/researcher/.env
mc alias set minio http://localhost:9000 "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY"
mc cp /tmp/book.epub "minio/$MINIO_BUCKET/books/<slug>.epub"
mc cp /tmp/book_full.txt "minio/$MINIO_BUCKET/books/<slug>.txt"
```

## Public URL for notes

Once uploaded, the file is accessible at:
```
http://vmi3304846.tail5c02a1.ts.net:9000/knowledge-base/books/<slug>.epub
http://vmi3304846.tail5c02a1.ts.net:9000/knowledge-base/books/<slug>.txt
```

Include this in the note frontmatter:
```yaml
source_file: http://vmi3304846.tail5c02a1.ts.net:9000/knowledge-base/books/<slug>.epub
```

## Folder mapping

| File type | MinIO path | Frontmatter field |
|-----------|-----------|-------------------|
| Books (ePub/PDF) | `books/<slug>.<ext>` | `source_file` |
| Videos (YouTube) | `videos/<slug>.webm` | `source_file` |
| Videos (Reels) | `reels/<slug>.mp4` | `source_file` |
| Transcripts | `videos/<slug>.json` | `source_file` |
| Articles/Threads | `articles/<slug>.txt` | `source_file` |

All book files (original ePub/PDF + extracted full text) go under `books/` with the same slug.

## Notes

- mc alias is already configured on the server — skip step 1 on first use
- Bucket is set to `download` policy (public read) but only accessible via Tailscale network
- No authentication needed for downloads
