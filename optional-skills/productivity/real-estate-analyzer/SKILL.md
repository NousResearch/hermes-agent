---
name: real-estate-analyzer
description: "Analyze any property listing URL — extract details, compare with local market, and deliver buy/pass verdict."
tags: [real-estate, property, investment, analysis, browser]
platforms: [linux, macos, windows]
---
# Real Estate Analyzer

Analyze any property listing from any website. Give Hermes a listing URL and it will extract the property details directly from the page, find comparable listings in the same area, and deliver a structured market analysis with a clear verdict.

No API keys required — uses browser and web_search only.

## When to Use
- User shares a property listing URL and asks for analysis
- User says "analyze this listing", "is this a good deal", "compare this to the market"
- User asks "should I buy this property"
- User wants to know if a price is fair for the area

## Site Compatibility
Works with any publicly accessible property listing page — no site-specific configuration needed. The browser tool reads whatever page the URL points to. Common examples: Sahibinden, Zingat, Zillow, Rightmove, Idealista, Immobilienscout24, SeLoger, Funda, Redfin. If the page requires login, Hermes will ask the user to paste the details manually.

## Analysis Workflow

### Step 1: Extract Listing Data
Open the listing URL with browser and extract:
- Price (total and per m²/sqft)
- Size (m² or sqft)
- Room count / bedrooms / bathrooms
- Location (city, district, neighborhood)
- Building age / year built
- Floor / total floors
- Listing date (how long on market)
- Key features (parking — note: covered/open/none, balcony, elevator, garden, heating type)
- Site amenities (pool, gym, security/doorman, playground, generator, green areas)
- Description highlights and any red flags in the text

### Step 2: Find Comparable Listings
Use web_search to find 3-5 similar properties in the same area:
- Same neighborhood or district
- Similar size (±20%)
- Similar room count
- Listed in the last 30-60 days

Search queries to use:
- "[neighborhood] [size]m² [rooms] [site]"
- "[city] [district] apartment for sale [size]"
- Use the same platform the listing is on for fair comparison

### Step 3: Calculate Market Position
- Compute average price per m² from comparables
- Calculate how much above or below market the listing is
- Note how long comparables stayed on market (demand signal)
- Flag if the listing has been on market unusually long (negotiation leverage)

### Step 4: Assess the Listing
Evaluate:
- **Price fairness**: vs comparable m² price in the same area
- **Size & layout**: m²/sqft, room count, floor plan efficiency
- **Parking**: covered garage vs open vs none — significant price factor in urban areas
- **Site amenities**: pool, gym, 24h security, playground — add premium vs standalone buildings
- **Building age**: newer buildings command premium; older may need renovation budget
- **Location quality**: proximity to transport, schools, amenities (search if needed)
- **Building condition signals**: age, floor, description language
- **Red flags**: vague description, no photos, price drops, long time on market
- **Positive signals**: recent renovation, below market, motivated seller language

### Step 5: Deliver Report

Always output in this format:

## Property Analysis: [address or listing ID]

**Platform**: [site name]
**Price**: [total price] ([price per m²/sqft])
**Size**: [m² or sqft] | **Rooms**: [N] | **Age**: [year or unknown]
**Location**: [neighborhood, city]

### Market Comparison
| Metric | This Listing | Area Average |
|--------|-------------|--------------|
| Price per m² | X | Y |
| Days on market | N | avg N |
| Total price | X | avg X |
| Parking | yes/no/type | — |

**Market Position**: [X% above / below / at market]

### Advantages
- [specific advantage]
- [specific advantage]

### Disadvantages / Red Flags
- [specific issue]
- [specific issue]

### Location Assessment
[1-2 sentences on neighborhood quality, transport, amenities]

### Verdict
**[BUY / NEGOTIATE / PASS]**

[2-3 sentence justification. If NEGOTIATE, suggest a target price.]

## Key Principles
- Always open the actual listing page with browser — never guess details
- Find real comparables, not just any listings in the city
- Price per m² is the primary comparison metric
- Long time on market (60+ days) is leverage for negotiation
- Vague or missing details in description are red flags
- Always state confidence level if comparable data is thin
- If the listing requires login to view, say so and ask user to paste the details manually
