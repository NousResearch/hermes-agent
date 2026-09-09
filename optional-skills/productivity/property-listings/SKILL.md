---
name: property-listings
description: Present property and rental listings as desktop cards.
version: 0.1.0
author: Teknium (teknium1), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [property, rental, real-estate, listings, desktop, cards]
    category: productivity
    related_skills: []
---

# Property Listings Skill

Present researched properties as browsable cards in the Hermes desktop transcript.
This is a presentation recipe, not a listing search service or an investment valuation.

## When to Use

- Presenting property or rental search results, comparing a shortlist, or re-ranking properties.
- Following up on a property already shown: keep using cards so the shortlist stays comparable.
- Outside the desktop app, use ordinary Markdown with source links instead; other clients need not render listing fences.

## Prerequisites

- A Hermes desktop conversation for native cards; the backend may be local or remote.
- Property details supplied by the user or verified through `web_search`, `web_extract`, or the browser tools available in this session.
- No additional API keys or dependencies are required for card formatting.

## How to Run

Install this optional skill through the Skills catalog, or use `terminal`:

```text
hermes skills install official/productivity/property-listings
```

Load it with `skill_view(name="property-listings")` when presenting listings.
Installing does not retrofit the running conversation's skill index; start a new
conversation for automatic discovery, or explicitly load the installed skill now.

## Quick Reference

Emit a fenced code block whose language is `listing` and whose body is valid JSON.
Use one object, an array of objects, or `{ "listings": [...] }` for a comparison.

| Field | Shape and meaning |
|---|---|
| `address` | Required nonempty street address or property headline. |
| `price` | Formatted string including currency and rental period, if applicable. |
| `beds`, `baths` | Positive numeric counts; omit unknown values. |
| `size` | Formatted area including units. |
| `note` | Why this property is worth a look. |
| `facts` | Array of short verified specs or amenities. |
| `catches` | Array of risks or questions to verify before a tour. |
| `images` | Direct HTTPS photo URLs in listing order; the first is the hero. |
| `links` | Array of `{ "label": "Source", "url": "https://..." }` detail-page links, not search-result URLs. |

## Procedure

1. Gather the address, price, specs, photos and canonical detail URL. Distinguish
   verified facts from unknowns; do not invent prices, amenities, or photo URLs.
2. Deduplicate portal mirrors of the same property into one card, retaining useful
   source links. Keep source dates and availability caveats in the surrounding prose.
3. Emit the `listing` fence for every property presented, including follow-ups and
   re-rankings. Keep facts short and put unresolved concerns in `catches`.
4. Check the JSON before sending. This fictional format example illustrates all fields;
   replace its values and example URLs with verified listing data:

```listing
{
  "address": "12 Example Lane",
  "price": "$2,400/mo",
  "beds": 3,
  "baths": 2.5,
  "size": "1,600 sqft",
  "note": "Fits the requested space and budget.",
  "facts": ["12-month lease", "Covered parking"],
  "catches": ["Verify pet policy and total move-in fees"],
  "images": ["https://example.com/property/front.jpg", "https://example.com/property/kitchen.jpg"],
  "links": [{"label": "Listing details", "url": "https://example.com/property/12"}]
}
```

## Pitfalls

- Cards are authored from gathered data, not fetched from a listing URL or embedded portal page.
- A sparse card needs only an address. Omit unknown fields rather than filling them with guesses.
- Use direct remote image URLs, not local paths, data URLs, or search-result pages.
  Expired or blocked images disappear from the gallery; the text and links still matter.
- Keep a fence to at most 24 properties, 40 images per property, and 12 entries in
  facts, catches and links. Text fields are truncated to 400 characters by the renderer.
- Malformed JSON or a card without identity falls back to a plain code block.
  A valid card is not proof that the underlying listing is current or accurate.

## Verification

- Every presented property has an address and a verified source link; unknowns are explicit.
- Desktop displays the address, price, specs, facts, catches and links as a native card.
- Photos form a gallery; selecting a photo opens the lightbox. Three or more photos
  use a hero-and-supporting-frames mosaic; additional photos remain browsable there.
- If the card fails to render, validate the fence language and JSON, then preserve a
  readable Markdown fallback with the same facts and links.
