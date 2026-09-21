---
name: amazon-search
description: >
  Search Amazon products with filters (department, brand, rating, price, Prime).
  Returns structured results with title, price, rating, reviews, and URL.
category: browser
---

# Amazon Search

## Target
- URL: https://www.amazon.com/s?k=<search_term>&<filters>
- Method: Browser (requires JS rendering + handles bot detection)

## Steps

1. **Navigate to search**
   ```
   browser_navigate("https://www.amazon.com/s?k=<search_term>")
   ```

2. **Wait for results**
   ```
   browser_snapshot(full=true)
   ```
   Amazon loads results dynamically.

3. **Apply filters** (optional, via URL params):
   - `rh=p_36:<min_price>-<max_price>` — price range (in cents)
   - `rh=p_72:<min_rating>` — star rating (e.g., p_72:2663393011 = 4+ stars)
   - `rh=p_85:<brand_id>` — specific brand
   - `prime=true` — Prime eligible

4. **Extract product data** from each result card:
   - `title` — product name
   - `price` — current price (with currency)
   - `rating` — star rating (e.g., 4.5)
   - `reviews` — number of reviews
   - `prime` — Prime eligibility (boolean)
   - `url` — product page link
   - `image` — thumbnail URL

5. **Return JSON**
   ```json
   {
     "query": "wireless headphones",
     "filters": {
       "price_min": 50,
       "price_max": 200,
       "rating_min": 4,
       "prime": true
     },
     "results": [
       {
         "title": "Sony WH-1000XM5",
         "price": "$298.00",
         "rating": 4.7,
         "reviews": 12847,
         "prime": true,
         "url": "https://www.amazon.com/dp/...",
         "image": "https://m.media-amazon.com/images/I/..."
       }
     ]
   }
   ```

## Pitfalls
- Amazon has aggressive bot detection; use delays between requests
- CAPTCHAs may appear; use `browser_vision` to detect
- Price format varies ($, USD, etc.)
- Some results are sponsored; note in output
- Search results vary by region/account

## Verification
1. Search for a known product
2. Verify price is reasonable
3. Check if filters are applied correctly
