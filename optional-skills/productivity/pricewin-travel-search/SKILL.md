---
name: pricewin-travel-search
description: Compare live hotel and flight prices via PriceWin.
version: 1.0.0
author: PriceWin
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Travel, Hotels, Flights, Booking, Price-Comparison]
    homepage: https://www.price.win/
    category: productivity
---

# PriceWin travel search

Live hotel and flight prices compared across Booking.com, Agoda, Trip.com and
Traveloka, plus direct-listed OpenTravel properties you can book end to end.

## Prerequisite

The tools come from the `pricewin` MCP catalog entry — no account, no API key:

```bash
hermes mcp install pricewin
```

Tools appear prefixed: `mcp_pricewin_search_hotels_live`,
`mcp_pricewin_poll_search_results`, and so on. The names below omit the prefix.

Booking and cancellation tools are **off by default**. Turn them on only if the
user wants to book through the conversation: `hermes mcp configure pricewin`.

## Results arrive in two steps, always

`search_hotels_live` opens a session and returns almost immediately with few or
no results. `poll_search_results` returns what has landed since. This is not an
error state — it is how the product works.

```
search_hotels_live(city="Da Nang", checkIn="2026-10-02", checkOut="2026-10-05", adults=2)
  → sessionId

poll_search_results(sessionId=…, nights=3)        # nights is REQUIRED
  → partial list, plus whether more is coming
```

Poll every few seconds until the result set stops growing or the user has
enough to choose from. Show the best options you already have while polling
rather than waiting in silence — prices for the top properties usually settle
first. If the user asks why more keep appearing, say the search is still
running; do not describe the mechanics behind it.

Flights work the same way: `search_flights_live` → `poll_flight_results`
(sessionId only, no `nights`).

## Do not invent what the user did not say

Every search parameter is optional on purpose. If the user has not named a
city, a date, or a party size, **omit the argument** — the tool replies with
what to ask for. Filling in a plausible city or "tomorrow" produces a confident
answer about the wrong trip.

The same goes for the optional filters. Pass `hotelName` only when the user
named a property and `area` only when they named a district — never the city
again, never an empty string or "all" to fill the slot. The server ignores
those, but any other filler turns a city-wide search into a name lookup the
guest never asked for.

Ask before searching when the user gave none of: destination, dates, number of
guests. Dates are the user's local dates — resolve "this weekend" against their
timezone, not the server's.

## Prices are USD, budgets cover the whole stay

Every price the tools return is USD. `priceMin` / `priceMax` filter on the
**total stay**, not per night: a $60/night ceiling over 3 nights is
`priceMax=180`. Set `priceCurrency` when the user stated a budget in something
else (`priceCurrency="VND"`, `"EUR"`, …) and the filter is converted for you.

## Answer in the guest's language

Pass `queryText` with a verbatim excerpt of the user's own request. Language
follows the words the guest actually typed — not your own locale, not a
previous turn's guess. Only fall back to `language` when `queryText` carries no
signal.

## Inline cards in Hermes Desktop

If the `pricewin` **plugin** is installed, put each recommendation on its own
paragraph as a directive and the transcript renders it as a card:

```
::pricewin-hotel{name="Liberty Central Riverside" price="58" stars="4" area="District 1, Ho Chi Minh City" ota="agoda" link="agoda.com/liberty-central/hotel/ho-chi-minh-city-vn.html?checkIn=2026-10-20&los=2&adults=2"}

::pricewin-flight{route="SGN → HAN" airline="Vietnam Airlines" price="72" depart="06:15" duration="2h10m" stops="0"}
```

Rules: the directive must be the whole paragraph, on one line. `price` is a bare
USD number — **per night** for a hotel (divide the stay total by the nights), the
fare total for a flight. For `ota` and `link`, copy the `card: ota=… link="…"`
that `poll_search_results` prints after each hotel, exactly as given. Never put
a URL with `https://` or `www.` in a directive: Desktop turns it into a hyperlink
and the card is shown as raw text. With no link for a result, leave `link` out.
With no plugin installed the line renders as ordinary text, so it is safe to emit
either way — but do not emit more than the three to five options worth showing.

## Detail before booking

- `get_ota_hotel_detail(hotelName, city, checkIn, checkOut)` — rooms, photos,
  facilities and reviews for one named property on the OTAs. Pass `propertyUrl`
  from a prior result to skip re-resolving the name.
- `get_hotel_detail` / `get_hotel_info` — direct-listed OpenTravel properties,
  the only ones bookable in-conversation.
- `get_cancellation_policy` — fetch the real policy before the guest commits.

## Booking (opt-in tools)

`request_booking` sends a booking **request** to the hotel. Nothing is charged
and no room is held — the hotel confirms the request, and the guest pays at the
property on arrival. There is no payment step anywhere, so never ask for card
details and never promise a payment link. Use `check_booking_status` when the
guest asks what became of a request.

Leave out guest details the guest has not given you: the tool answers with the
question to put to them rather than inventing a value.

Cancelling is two steps and cannot be done silently: `request_cancel_token`
emails the guest a single-use token, they paste it back, then `cancel_booking`
with confirmation code, token and reason. It is irreversible — read the refund
terms back to the guest first.
