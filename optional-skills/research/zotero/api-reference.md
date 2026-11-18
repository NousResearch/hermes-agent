# Zotero API Reference

Base URL: `https://api.zotero.org/users/{userID}`  
Headers (all requests): `Zotero-API-Version: 3`, `Zotero-API-Key: {key}`

---

## Collections

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/collections` | All collections (flat list) |
| `GET` | `/collections/top` | Top-level collections only |
| `GET` | `/collections/{key}` | Single collection metadata |
| `GET` | `/collections/{key}/collections` | Subcollections |
| `GET` | `/collections/{key}/items` | All items in collection |
| `GET` | `/collections/{key}/items/top` | Top-level items (no children) |
| `POST` | `/collections` | Create collection(s) |
| `PUT` | `/collections/{key}` | Replace collection |
| `PATCH` | `/collections/{key}` | Update collection fields |
| `DELETE` | `/collections/{key}` | Delete collection |

**Create collection:**
```json
POST /collections
[{"name": "Books", "parentCollection": "PARENT_KEY"}]
```
Response: `200 OK` with `{"successful": {"0": {"key": "NEWKEY", ...}}}`

**Top-level collections response example:**
```json
[
  {"key": "ABCD1234", "version": 42, "data": {"key": "ABCD1234", "name": "hermes-agent", "parentCollection": false}},
  {"key": "EFGH5678", "version": 43, "data": {"key": "EFGH5678", "name": "Books", "parentCollection": "ABCD1234"}}
]
```

---

## Items

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/items` | All items in library |
| `GET` | `/items/top` | Top-level items only |
| `GET` | `/items/{key}` | Single item |
| `GET` | `/items/{key}/children` | Child items (notes, attachments) |
| `POST` | `/items` | Create item(s) (up to 50) |
| `PUT` | `/items/{key}` | Replace item (requires version) |
| `PATCH` | `/items/{key}` | Partial update |
| `DELETE` | `/items/{key}` | Delete item |
| `DELETE` | `/items?itemKey=K1,K2` | Batch delete |

**Item query parameters:**

| Param | Description | Example |
|-------|-------------|---------|
| `q` | Keyword search | `q=deep+learning` |
| `qmode` | Search mode: `titleCreatorYear` or `everything` | `qmode=everything` |
| `itemType` | Filter by type | `itemType=book` |
| `tag` | Filter by tag | `tag=unread` |
| `collection` | Filter by collection key | `collection=ABCD1234` |
| `since` | Library version (for sync) | `since=42` |
| `limit` | Results per page (1-100, default 25) | `limit=100` |
| `start` | Pagination offset | `start=100` |
| `sort` | Sort field | `sort=dateAdded` |
| `direction` | `asc` or `desc` | `direction=desc` |

**Pagination:** Check `Total-Results` response header. Use `start` to page through.

**Create journal article:**
```json
POST /items
[{
  "itemType": "journalArticle",
  "title": "Attention Is All You Need",
  "creators": [{"creatorType": "author", "firstName": "Ashish", "lastName": "Vaswani"}],
  "abstractNote": "...",
  "publicationTitle": "NeurIPS",
  "volume": "30",
  "date": "2017",
  "DOI": "10.48550/arXiv.1706.03762",
  "url": "https://arxiv.org/abs/1706.03762",
  "tags": [{"tag": "unread"}, {"tag": "transformers"}],
  "collections": ["COLLECTION_KEY"]
}]
```

**Create book:**
```json
[{
  "itemType": "book",
  "title": "The Pragmatic Programmer",
  "creators": [{"creatorType": "author", "firstName": "David", "lastName": "Thomas"}],
  "publisher": "Addison-Wesley",
  "date": "2019",
  "ISBN": "978-0-13-595705-9",
  "numPages": "352",
  "collections": ["BOOKS_COLLECTION_KEY"]
}]
```

**Create webpage item:**
```json
[{
  "itemType": "webpage",
  "title": "Page Title",
  "url": "https://example.com",
  "websiteTitle": "Site Name",
  "accessDate": "2026-03-24",
  "collections": ["COLLECTION_KEY"]
}]
```

**Item types reference:**

| Type | `itemType` value |
|------|-----------------|
| Journal Article | `journalArticle` |
| Book | `book` |
| Book Section | `bookSection` |
| Conference Paper | `conferencePaper` |
| Thesis | `thesis` |
| Preprint | `preprint` |
| Webpage | `webpage` |
| Report | `report` |
| Magazine Article | `magazineArticle` |
| Blog Post | `blogPost` |

---

## Notes

Notes are items with `itemType: "note"` and a `parentItem` key.

**Create a child note:**
```json
POST /items
[{
  "itemType": "note",
  "parentItem": "PARENT_ITEM_KEY",
  "note": "<p><strong>Summary</strong></p><p>Key points here...</p>",
  "tags": [{"tag": "reading-note"}]
}]
```

The `note` field is **HTML**. Plain text should be wrapped in `<p>` tags.

**Update a note (PATCH):**
```
PATCH /items/{noteKey}
If-Unmodified-Since-Version: {current_version}

{"note": "<p>Updated content</p>", "version": {current_version}}
```

---

## Attachments & PDFs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/items/{key}/children` | Get child items (find attachment key) |
| `GET` | `/items/{attachmentKey}/file` | Download the file (follows redirect) |
| `GET` | `/items/{attachmentKey}/fulltext` | Get indexed text content |
| `PUT` | `/items/{attachmentKey}/fulltext` | Set/update indexed text |

**Fulltext response:**
```json
{
  "content": "Extracted text from the PDF...",
  "indexedPages": 12,
  "totalPages": 42
}
```

**Download PDF:** The `/file` endpoint returns a redirect to the actual file (Zotero cloud or WebDAV). Follow redirects with `requests.get(..., allow_redirects=True)`.

**Attachment item fields:**
```json
{
  "itemType": "attachment",
  "linkMode": "imported_file",
  "title": "paper.pdf",
  "filename": "paper.pdf",
  "contentType": "application/pdf",
  "md5": "abc123...",
  "mtime": 1700000000000
}
```

---

## Tags

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/tags` | All tags in library |
| `GET` | `/items/{key}/tags` | Tags on one item |
| `GET` | `/collections/{key}/tags` | Tags in a collection |
| `DELETE` | `/tags?tag=name` | Delete tag from all items |

Tags are set on items in the `tags` array field:
```json
"tags": [{"tag": "unread"}, {"tag": "machine-learning"}]
```

---

## Versioning & Writes

Every item/collection has a `version` integer. Writes must include:
- `If-Unmodified-Since-Version: {version}` header (single-item writes)
- `version` in the JSON body (PATCH)

For new items (`POST`), no version is needed. Response includes `Last-Modified-Version` header with the new library version.

**Write token** (idempotency for `POST`):
```
Zotero-Write-Token: {random-32-char-hex}
```

---

## External Metadata APIs

### CrossRef (DOI lookup)
```
GET https://api.crossref.org/works/{DOI}
```
Key response fields: `title[0]`, `author[].given/family`, `published.date-parts[0]`, `container-title[0]`, `DOI`, `URL`, `type`

### Open Library (ISBN lookup)
```
GET https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&format=json&jscmd=data
```
Key response fields (under `ISBN:{isbn}`): `title`, `authors[].name`, `publishers[].name`, `publish_date`, `number_of_pages`, `identifiers.isbn_13[0]`

### arXiv (arXiv ID lookup)
```
GET https://export.arxiv.org/api/query?id_list={arxiv_id}
```
Returns Atom XML. Parse namespace `http://www.w3.org/2005/Atom`. Key tags: `title`, `author/name`, `published`, `summary`, `id` (URL containing the ID).

### Google Books (ISBN fallback)
```
GET https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}
```
Key fields: `items[0].volumeInfo.{title, authors[], publisher, publishedDate, pageCount, industryIdentifiers[]}`

---

## Error Responses

| Status | Meaning |
|--------|---------|
| `200` | Success (reads + some writes) |
| `204` | Success, no body (deletes) |
| `400` | Bad request / malformed JSON |
| `403` | Invalid API key or insufficient permissions |
| `404` | Item/collection not found |
| `409` | Version conflict (`If-Unmodified-Since-Version` mismatch) |
| `412` | Precondition failed |
| `429` | Rate limit exceeded — back off and retry |

---

## Zotero Item Key Format

Keys are 8-character uppercase alphanumeric strings, e.g. `A3BC4DEF`.
