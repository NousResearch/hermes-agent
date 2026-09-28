# Python SDK — ContentContext

`ContentContext` (sync) / `AsyncContentContext` (async) — news and discussion topics for securities.

## Creation

```python
from longbridge.openapi import ContentContext, AsyncContentContext, Config

# Sync
ctx = ContentContext(config)

# Async
ctx = AsyncContentContext.create(config)
```

## Methods

### news

Get news articles for a symbol.

```python
items = ctx.news("700.HK")   # List[NewsItem]
# Fields: id, title, description, url, published_at,
#         likes_count, comments_count, shares_count
```

### topics

Get discussion topics (community posts) for a symbol.

```python
items = ctx.topics("700.HK")   # List[TopicItem]
# Fields: id, title, description, url, published_at,
#         likes_count, comments_count, shares_count
```

## Async Example

```python
import asyncio
from longbridge.openapi import AsyncContentContext, Config

async def main():
    config = Config.from_apikey_env()
    ctx = AsyncContentContext.create(config)

    news = await ctx.news("700.HK")
    for item in news:
        print(item.title, item.published_at)

    topics = await ctx.topics("AAPL.US")
    for item in topics:
        print(item.title, item.likes_count)

asyncio.run(main())
```

## Sync Example

```python
from longbridge.openapi import ContentContext, Config

config = Config.from_apikey_env()
ctx = ContentContext(config)

news = ctx.news("700.HK")
for item in news:
    print(item.title, item.url)
```
