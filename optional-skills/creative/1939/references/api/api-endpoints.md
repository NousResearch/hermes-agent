# 1939 API Endpoints

Base URL: `https://1939.cuttle.af`

## Themes

```
GET /api/themes              → all 29 themes (slim by default)
GET /api/themes?detail=full → all 29 themes with pipeline data
GET /api/themes/{slug}      → single theme by slug
```

## Cards

```
GET /api/cards                      → all 496 cards (slim, paginated)
GET /api/cards?season=1             → cards from season 1
GET /api/cards?collection=6529-memes → filter by collection
GET /api/cards/{slug}               → single card
GET /api/cards/{slug}?detail=full   → full data including pipeline
```

## Palette Details

```
GET /api/palettes/{slug}/roles     → 8-role definitions with tints
GET /api/palettes/{slug}/spectrum  → 15-color perceptual spectrum
GET /api/palettes/{slug}/contrast  → 7 WCAG contrast ratios
```

## Collections & Metadata

```
GET /api/collections   → collection info (1939-flagship, 6529-memes)
GET /api/seasons        → season aggregates (1-15)
GET /api/benchmarks     → pipeline aggregation stats
GET /api/health         → service status
```

## Query Parameters

| Param | Values | Default |
|-------|--------|---------|
| `detail` | `slim`, `full` | `slim` |
| `collection` | `1939-flagship`, `6529-memes` | all |
| `season` | `1`–`15` | all |
| `fields` | comma-separated field list | all |
| `sort` | `id`, `pv`, `year`, `name` | `id` |
| `order` | `asc`, `desc` | `asc` |
| `limit` | 1–200 | 50 |
| `offset` | integer | 0 |

**Slim response** strips: pipeline, pool, raw_hexes, filtered_hexes, attributes, dataPalette, contrast, meta, img, volume, tags.

**Full response** includes everything.

## Response Headers

- `X-Data-Version`: Data version date (e.g., `2026-05-24`)
- `X-RateLimit-Limit`: 100 requests per minute
- `Content-Type`: `application/json`

## Rate Limiting

100 requests per minute per IP. Returns `429 Too Many Requests` when exceeded.

## Swagger Docs

Interactive API documentation: `https://1939.cuttle.af/docs`