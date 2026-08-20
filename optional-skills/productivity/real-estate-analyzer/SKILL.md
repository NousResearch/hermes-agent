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
Use web_search to find as many comparable listings as available in the same area (aim for at least 3, more is better for accuracy):
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
**Investment Score: X/10**

[2-3 sentence justification. If NEGOTIATE, suggest a target price.]

Score breakdown:
- Price vs market: X/4
- Features & amenities: X/3
- Location & demand: X/2
- Risk deductions: -X


## Multi-Listing Comparison

When user shares 2-3 listing URLs, compare them side by side:

1. Extract data from each listing using browser
2. Find comparables for each in the same area
3. Calculate Investment Score for each (see below)
4. Output a unified comparison table with winner recommendation

### Multi-Listing Output Format

    ## Listing Comparison

    | Metric | Listing 1 | Listing 2 | Listing 3 |
    |--------|-----------|-----------|-----------|
    | Price | X | Y | Z |
    | Price/m² | X | Y | Z |
    | Size | X | Y | Z |
    | Rooms | X | Y | Z |
    | Age | X | Y | Z |
    | Parking | X | Y | Z |
    | Site amenities | X | Y | Z |
    | Days on market | X | Y | Z |
    | vs Area avg | X% | Y% | Z% |
    | Investment Score | X/10 | Y/10 | Z/10 |
    | Verdict | BUY | NEGOTIATE | PASS |

    **Winner**: Listing N — [1-2 sentence justification comparing the options]

## Investment Score

In addition to BUY / NEGOTIATE / PASS, always calculate an **Investment Score (1-10)**:

| Score | Meaning |
|-------|---------|
| 8-10 | Excellent deal — act fast |
| 6-7 | Good value — worth pursuing |
| 4-5 | Fair market price — negotiate hard |
| 2-3 | Overpriced — significant discount needed |
| 1 | Pass — not worth pursuing |

Score components:
- **Price vs market (40%)**: >15% below market = +4, 5-15% below = +3, at market = +2, above market = +1
- **Features & amenities (30%)**: covered parking, site amenities (pool/gym/security), building condition
- **Location & demand (20%)**: transport links, schools, walkability, days on market signal
- **Risk deductions (10%)**: vague description (-1), no photos (-1), price drop history (-1), 60+ days on market (-1)

Always show the score breakdown alongside the verdict.


## Example Output

### Single Listing

    ## Property Analysis: 3+1 Flat, Kadıköy, Istanbul

    **Platform**: Sahibinden
    **Price**: $285,000 ($3,187/m²)
    **Size**: 89m² | **Rooms**: 3+1 | **Age**: 2015
    **Location**: Moda, Kadıköy, Istanbul

    ### Market Comparison
    | Metric | This Listing | Area Average |
    |--------|-------------|--------------|
    | Price per m² | $3,187 | $3,450 |
    | Days on market | 18 | avg 35 |
    | Total price | $285,000 | avg $307,050 |
    | Parking | covered | — |

    **Market Position**: 7.6% below market

    ### Advantages
    - Priced 7.6% below comparable Moda listings
    - Covered parking — rare in this district, adds ~$12,000 value
    - Only 18 days on market — seller likely motivated, not distressed
    - 2015 build — no major renovation costs expected for 5-10 years

    ### Disadvantages / Red Flags
    - No gym or pool in the complex
    - 4th floor out of 8 — mid-floor, no discount or premium

    ### Location Assessment
    Moda offers excellent walkability, ferry access to the European side,
    and strong rental demand from young professionals. Appreciation trend positive.

    ### Verdict
    **NEGOTIATE**
    **Investment Score: 7/10**

    Offer $270,000 (5% below asking) — already 7.6% below market with covered
    parking. Comparable demand suggests seller will accept within 2 weeks.
    Strong rental yield potential at ~4.2% annual gross.

    Score breakdown:
    - Price vs market: 3/4 (7.6% below)
    - Features & amenities: 2.5/3 (covered parking, no pool/gym)
    - Location & demand: 1.8/2 (high demand area, fast moving)
    - Risk deductions: -0.3 (no pool/gym minor deduction)

### Multi-Listing Comparison

    ## Listing Comparison

    | Metric | Listing 1 (Kadıköy) | Listing 2 (Beşiktaş) | Listing 3 (Şişli) |
    |--------|---------------------|----------------------|-------------------|
    | Price | $285,000 | $320,000 | $265,000 |
    | Price/m² | $3,187 | $3,764 | $3,312 |
    | Size | 89m² | 85m² | 80m² |
    | Rooms | 3+1 | 3+1 | 2+1 |
    | Age | 2015 | 2008 | 2020 |
    | Parking | covered | none | open |
    | Site amenities | none | gym | pool, gym |
    | Days on market | 18 | 47 | 5 |
    | vs Area avg | -7.6% | +4.2% | -2.1% |
    | Investment Score | 7/10 | 4/10 | 6/10 |
    | Verdict | NEGOTIATE | PASS | CONSIDER |

    **Winner**: Listing 1 (Kadıköy) — best price vs market, covered parking,
    strong location. Listing 2 is overpriced and stale. Listing 3 is newer
    but smaller and just listed, worth monitoring.

## Key Principles
- Always open the actual listing page with browser — never guess details
- Find real comparables, not just any listings in the city
- Price per m² is the primary comparison metric
- Long time on market (60+ days) is leverage for negotiation
- Vague or missing details in description are red flags
- Always state confidence level if comparable data is thin
- If the listing requires login to view, say so and ask user to paste the details manually
