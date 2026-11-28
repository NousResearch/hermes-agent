# Benchmarking HTML Report Template

Revelata dark-theme benchmark report. The design is **data-forward**: the primary element is a
fingerprint × coverage grid (KPIs as rows, companies as columns), not a prose fingerprint box.
Benchmark cards below the grid provide narrative context per company.

**Logo:** The Revelata wordmark is **inlined as SVG** in the header (same asset as Analysis Pressure
Test reports). No external image file. The logo link targets `https://www.revelata.com/for-ai-builders`.
**Header layout:** `.header` is a **single-column stack** — `.header-brand` row with SVG,
pipe divider, **Benchmark Analysis (GitHub)** link (no separate product label in that row);
then `h1` with cyan `.ticker` span; then `.header-meta`. **No legend in the header.**
**Legend placement:** `.legend-strip` is a **standalone** horizontal flex row placed **immediately
below** the fingerprint grid (after `</div><!-- grid-wrap -->`), not inside `.header`.
**Fonts:** Inter + Figtree (Google Fonts `@import`) plus JetBrains Mono for numeric cells.

---

## Design principles for benchmarking reports

- Fingerprint × coverage grid as hero
- **No commentary or prose inside the grid.** The hero table is **data only**: group-header rows,
  column headers, KPI label rows, numeric cells (with provenance links), and `—` for missing
  coverage. Do **not** add rows such as "Why it matters", thesis blurbs, `fp-context`, or any
  explanatory copy inside `table.fp-grid`. Put all narrative in **Benchmark Detail** cards and
  **Diff Insight** below the grid.
- Plain category dividers (lighter font, no tier numbers)
- **Tickers only** in column headers
- **White** = target column (amber/yellow means partial match)
- Corrected values shown silently
- `.bmark-row` compact row layout

**Ticker color coding (both column headers and card headings):**
- White `#fff` — target company
- Green `var(--green)` `#22c55e` — strong structural match
- Yellow `var(--yellow)` `#eab308` — partial match
- Purple `var(--purple)` `#a855f7` — segment sub-benchmark
- No "red" tickers — companies with no match simply aren't included

**Cell color coding in the grid:**
- `td.cell.target-col` — `background: transparent; color: #fff` (target text reads white; row hover adds a light wash — see **Fingerprint grid** CSS)
- `td.cell.match-strong` — `background: var(--green-mid); color: #86efac`
- `td.cell.match-partial` — `background: var(--yellow-mid); color: #fde68a`
- `td.cell.match-none` — transparent, `color: var(--text-dim)`, shows `—` or brief note
- `td.cell.seg-cell` — `background: var(--purple-dim); color: #d8b4fe`

---

## CSS variables (do not change)

```css
:root {
  --bg: #0a0d13; --surface: #111318; --surface2: #181c24;
  --border: #252b36; --border-light: #2e3647;
  --text: #dce4f0; --text-muted: #7a8799; --text-dim: #4e5a6a;
  --cyan: #4FEAFF; --cyan-dim: rgba(79,234,255,0.1); --cyan-glow: rgba(79,234,255,0.06);
  --amber: #f59e0b; --amber-dim: rgba(245,158,11,0.1); --amber-text: #fbbf24;
  --green: #22c55e; --green-dim: rgba(34,197,94,0.12); --green-mid: rgba(34,197,94,0.2);
  --yellow: #eab308; --yellow-dim: rgba(234,179,8,0.12); --yellow-mid: rgba(234,179,8,0.18);
  --red: #ef4444; --red-dim: rgba(239,68,68,0.1);
  --purple: #a855f7; --purple-dim: rgba(168,85,247,0.12);
  --blue: #3b82f6; --blue-dim: rgba(59,130,246,0.1);
}
```

---

## Page shell: fonts, body, and header (match Analysis Pressure Test)

The **top of every benchmark HTML file** uses the same `body` canvas as Analysis Pressure Test
(`padding: 2.5rem`, `max-width: 900px`, `font-size: 14.5px`). The **header chrome** matches the
pressure-test asset and links: **inlined Revelata SVG** in an anchor to
`https://www.revelata.com/for-ai-builders`, pipe divider, and **Benchmark Analysis (GitHub)** —
all inline in `.header-brand`. The header is a **single-column stack** (no legend inside it). The
color legend is a separate **`.legend-strip`** placed **after the fingerprint grid** (see **Grid skeleton** below).

Use the **complete** last `<path>` for the coin mark (inner strokes) — truncated copies omit detail
and look wrong. Title line: `h1` with `<span class="ticker">[TICKER]</span>` in cyan; metadata:
`.header-meta` with optional `.meta-dot` on the first item.

Merge this **after** the benchmark `:root` block. `.header h1` and `.header-meta` are scoped under
`.header` so they do not fight other headings if you add section titles later.

```css
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Figtree:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Inter', system-ui, sans-serif;
  background: var(--bg); color: var(--text);
  line-height: 1.7; padding: 2.5rem;
  max-width: 900px; margin: 0 auto;
  font-size: 14.5px; -webkit-font-smoothing: antialiased;
}

/* Header — single-column stack, no legend */
.header {
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
  margin-bottom: 1rem;
}
.header-brand {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin-bottom: 0.75rem;
  flex-wrap: wrap;
}
.header-brand .header-logo {
  display: block;
  height: 22px;
  width: auto;
  flex-shrink: 0;
}
.header-brand > a {
  color: inherit;
  text-decoration: none;
  line-height: 0;
}
.header-brand > a:hover { opacity: 0.9; }
.header-divider {
  color: var(--border-light);
  font-weight: 300;
  font-size: 1.25rem;
  line-height: 1;
  margin: 0 0.05rem;
}
.header-github a {
  font-family: 'Figtree', 'Inter', system-ui, sans-serif;
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--cyan);
  text-decoration: none;
}
.header-github a:hover {
  text-decoration: underline;
  text-decoration-style: dotted;
  text-underline-offset: 3px;
}
.header h1 {
  font-size: 1.6rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  color: #fff;
  margin-bottom: 0.35rem;
}
.header h1 .ticker { color: var(--cyan); }
.header-meta {
  margin-top: 0.15rem;
  font-size: 0.78rem;
  color: var(--text-muted);
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  align-items: center;
}
.header-meta span {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
}
.meta-dot {
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--cyan);
  flex-shrink: 0;
}

/* Legend strip — standalone element BELOW the grid, not inside the header */
.legend-strip {
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
  gap: 1.4rem;
  align-items: center;
  font-size: 0.78rem;
  color: var(--text-muted);
  padding: 0.65rem 0.8rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  margin-top: 0.6rem;
  margin-bottom: 2.5rem;
}
.legend-strip .legend-item { display: flex; align-items: center; gap: 0.4rem; }
.legend-strip .legend-swatch {
  width: 10px;
  height: 10px;
  border-radius: 2px;
  flex-shrink: 0;
}

@media (max-width: 700px) {
  body { padding: 1.2rem; }
}

```

### Header HTML (embedded SVG — required)

```html
<!-- ===== HEADER ===== -->
<div class="header">
    <div class="header-brand">
      <a href="https://www.revelata.com/for-ai-builders" target="_blank" rel="noopener noreferrer" aria-label="Revelata">
        <svg class="header-logo" width="159" height="34" viewBox="0 0 159 34" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="Revelata Inc">
          <path d="M42.4412 26.6676V7.28626H48.7539C50.0091 7.28626 51.1166 7.53545 52.0765 8.03382C53.0548 8.5322 53.8116 9.22439 54.3468 10.1104C54.9006 10.9964 55.1775 12.0116 55.1775 13.156C55.1775 14.3005 54.8821 15.3157 54.2915 16.2017C53.7008 17.0877 52.8979 17.7891 51.8826 18.3059C50.8674 18.8043 49.723 19.0535 48.4494 19.0535H44.7946V26.6676H42.4412ZM53.3778 26.6676L48.7263 18.7213L50.5813 17.503L55.9527 26.6676H53.3778ZM44.7946 16.8939H49.0862C49.7876 16.8939 50.4152 16.737 50.969 16.4232C51.5227 16.0909 51.9657 15.6479 52.298 15.0942C52.6487 14.522 52.824 13.8759 52.824 13.156C52.824 12.0485 52.4364 11.1533 51.6611 10.4703C50.9043 9.76891 49.9168 9.4182 48.6986 9.4182H44.7946V16.8939ZM65.1997 26.9999C63.8523 26.9999 62.6617 26.6953 61.628 26.0862C60.5944 25.4586 59.7822 24.6003 59.1915 23.5112C58.6193 22.4222 58.3332 21.167 58.3332 19.7457C58.3332 18.3244 58.6193 17.0692 59.1915 15.9802C59.7822 14.8911 60.5851 14.042 61.6004 13.4329C62.634 12.8053 63.8154 12.4915 65.1444 12.4915C66.4365 12.4915 67.5532 12.8146 68.4946 13.4606C69.436 14.0882 70.1651 14.9742 70.6819 16.1186C71.1987 17.263 71.4572 18.6105 71.4572 20.161H60.0222L60.5759 19.6903C60.5759 20.7978 60.7697 21.7484 61.1574 22.5421C61.5634 23.3359 62.1264 23.945 62.8463 24.3695C63.5662 24.7756 64.3876 24.9787 65.3105 24.9787C66.2888 24.9787 67.1102 24.7479 67.7747 24.2865C68.4577 23.825 68.9837 23.2159 69.3529 22.4591L71.2633 23.4281C70.9126 24.148 70.4419 24.7756 69.8513 25.3109C69.2791 25.8462 68.5961 26.2615 67.8024 26.5569C67.0271 26.8522 66.1596 26.9999 65.1997 26.9999ZM60.7144 18.7489L60.1329 18.3059H69.6851L69.1037 18.7766C69.1037 17.8906 68.9283 17.1246 68.5776 16.4786C68.2269 15.8325 67.7562 15.3341 67.1656 14.9834C66.5749 14.6327 65.8919 14.4574 65.1167 14.4574C64.3599 14.4574 63.64 14.6327 62.957 14.9834C62.2925 15.3341 61.748 15.8325 61.3235 16.4786C60.9174 17.1061 60.7144 17.8629 60.7144 18.7489ZM78.1355 26.6676L83.7838 12.8238H86.1095L80.3228 26.6676H78.1355ZM77.9971 26.6676L72.2103 12.8238H74.5084L80.1567 26.6676H77.9971ZM94.023 26.9999C92.6755 26.9999 91.485 26.6953 90.4513 26.0862C89.4176 25.4586 88.6055 24.6003 88.0148 23.5112C87.4426 22.4222 87.1565 21.167 87.1565 19.7457C87.1565 18.3244 87.4426 17.0692 88.0148 15.9802C88.6055 14.8911 89.4084 14.042 90.4236 13.4329C91.4573 12.8053 92.6386 12.4915 93.9676 12.4915C95.2597 12.4915 96.3765 12.8146 97.3178 13.4606C98.2592 14.0882 98.9883 14.9742 99.5052 16.1186C100.022 17.263 100.28 18.6105 100.28 20.161H88.8454L89.3992 19.6903C89.3992 20.7978 89.593 21.7484 89.9806 22.5421C90.3867 23.3359 90.9497 23.945 91.6696 24.3695C92.3894 24.7756 93.2108 24.9787 94.1338 24.9787C95.1121 24.9787 95.9335 24.7479 96.598 24.2865C97.2809 23.825 97.807 23.2159 98.1762 22.4591L100.087 23.4281C99.7359 24.148 99.2652 24.7756 98.6745 25.3109C98.1023 25.8462 97.4194 26.2615 96.6256 26.5569C95.8504 26.8522 94.9828 26.9999 94.023 26.9999ZM89.5376 18.7489L88.9562 18.3059H98.5084L97.927 18.7766C97.927 17.8906 97.7516 17.1246 97.4009 16.4786C97.0502 15.8325 96.5795 15.3341 95.9888 14.9834C95.3982 14.6327 94.7152 14.4574 93.9399 14.4574C93.1832 14.4574 92.4633 14.6327 91.7803 14.9834C91.1158 15.3341 90.5713 15.8325 90.1467 16.4786C89.7407 17.1061 89.5376 17.8629 89.5376 18.7489ZM103.074 26.6676V7.28626H105.289V26.6676H103.074ZM117.686 26.6676L117.575 24.3972V19.3581C117.575 18.269 117.455 17.3738 117.215 16.6724C116.975 15.9525 116.597 15.408 116.08 15.0388C115.563 14.6696 114.889 14.485 114.058 14.485C113.302 14.485 112.646 14.6419 112.093 14.9557C111.557 15.2511 111.114 15.731 110.764 16.3955L108.77 15.6202C109.121 14.9742 109.545 14.4204 110.044 13.959C110.542 13.4791 111.124 13.1191 111.788 12.8792C112.453 12.6207 113.209 12.4915 114.058 12.4915C115.351 12.4915 116.412 12.75 117.243 13.2668C118.092 13.7652 118.728 14.5127 119.153 15.5095C119.578 16.4878 119.781 17.706 119.762 19.1643L119.734 26.6676H117.686ZM113.532 26.9999C111.908 26.9999 110.634 26.6307 109.712 25.8923C108.807 25.1356 108.355 24.0927 108.355 22.7636C108.355 21.3608 108.816 20.2902 109.739 19.5519C110.681 18.7951 111.991 18.4167 113.671 18.4167H117.63V20.2718H114.169C112.877 20.2718 111.954 20.484 111.4 20.9086C110.865 21.3331 110.598 21.9422 110.598 22.736C110.598 23.4558 110.865 24.028 111.4 24.4526C111.936 24.8587 112.683 25.0617 113.643 25.0617C114.437 25.0617 115.129 24.8956 115.72 24.5633C116.31 24.2126 116.763 23.7235 117.076 23.0959C117.409 22.4499 117.575 21.6931 117.575 20.8255H118.516C118.516 22.7083 118.092 24.2126 117.243 25.3386C116.393 26.4461 115.157 26.9999 113.532 26.9999ZM128.477 26.9999C127.13 26.9999 126.087 26.6491 125.349 25.9477C124.61 25.2463 124.241 24.2588 124.241 22.9851V8.50451H126.456V22.7636C126.456 23.4651 126.641 24.0096 127.01 24.3972C127.397 24.7664 127.933 24.951 128.616 24.951C128.837 24.951 129.049 24.9233 129.252 24.8679C129.474 24.7941 129.76 24.6279 130.111 24.3695L130.969 26.1692C130.489 26.483 130.055 26.6953 129.668 26.806C129.28 26.9352 128.883 26.9999 128.477 26.9999ZM121.832 14.7896V12.8238H130.609V14.7896H121.832ZM142.183 26.6676L142.072 24.3972V19.3581C142.072 18.269 141.952 17.3738 141.712 16.6724C141.472 15.9525 141.094 15.408 140.577 15.0388C140.06 14.6696 139.386 14.485 138.556 14.485C137.799 14.485 137.143 14.6419 136.59 14.9557C136.054 15.2511 135.611 15.731 135.261 16.3955L133.267 15.6202C133.618 14.9742 134.042 14.4204 134.541 13.959C135.039 13.4791 135.621 13.1191 136.285 12.8792C136.95 12.6207 137.706 12.4915 138.556 12.4915C139.848 12.4915 140.909 12.75 141.74 13.2668C142.589 13.7652 143.226 14.5127 143.65 15.5095C144.075 16.4878 144.278 17.706 144.259 19.1643L144.232 26.6676H142.183ZM138.029 26.9999C136.405 26.9999 135.132 26.6307 134.209 25.8923C133.304 25.1356 132.852 24.0927 132.852 22.7636C132.852 21.3608 133.313 20.2902 134.236 19.5519C135.178 18.7951 136.488 18.4167 138.168 18.4167H142.127V20.2718H138.666C137.374 20.2718 136.451 20.484 135.898 20.9086C135.362 21.3331 135.095 21.9422 135.095 22.736C135.095 23.4558 135.362 24.028 135.898 24.4526C136.433 24.8587 137.18 25.0617 138.14 25.0617C138.934 25.0617 139.626 24.8956 140.217 24.5633C140.807 24.2126 141.26 23.7235 141.573 23.0959C141.906 22.4499 142.072 21.6931 142.072 20.8255H143.013C143.013 22.7083 142.589 24.2126 141.74 25.3386C140.891 26.4461 139.654 26.9999 138.029 26.9999Z" fill="#ffffff"/>
          <path d="M146.357 28.6888V26.6676H157.432V28.6888H146.357Z" fill="#4FEAFF"/>
          <path d="M5.08739 7.94839L16.9844 2.21909L28.8814 7.94839L31.8197 20.822L23.5868 31.1459H10.3821L2.14907 20.822L5.08739 7.94839Z" stroke="#4FEAFF" stroke-width="1.78606"/>
          <path d="M7.16529 15.9562L12.0207 8.8345L20.616 8.19038L26.4787 14.5088L25.194 23.0319L17.7294 27.3416L9.70589 24.1926L7.16529 15.9562Z" stroke="#4FEAFF" stroke-width="1.78606"/>
          <path d="M13.3317 14.5225L16.9857 12.7628L20.6398 14.5225L21.5423 18.4765L19.0136 21.6474H14.9579L12.4292 18.4765L13.3317 14.5225Z" stroke="#4FEAFF" stroke-width="1.78606"/>
          <path d="M31.9318 30.2429C31.6471 30.2429 31.3754 30.1885 31.1166 30.0798C30.863 29.9711 30.6405 29.821 30.449 29.6295C30.2575 29.438 30.1074 29.2155 29.9987 28.9619C29.89 28.7031 29.8356 28.4314 29.8356 28.1467C29.8356 27.8517 29.89 27.5774 29.9987 27.3238C30.1074 27.0702 30.2575 26.8476 30.449 26.6561C30.6405 26.4646 30.863 26.3145 31.1166 26.2058C31.3754 26.0971 31.6471 26.0428 31.9318 26.0428C32.2268 26.0428 32.5011 26.0971 32.7547 26.2058C33.0084 26.3145 33.2309 26.4646 33.4224 26.6561C33.6139 26.8476 33.764 27.0702 33.8727 27.3238C33.9814 27.5774 34.0357 27.8517 34.0357 28.1467C34.0357 28.4314 33.9814 28.7031 33.8727 28.9619C33.764 29.2155 33.6139 29.438 33.4224 29.6295C33.2309 29.821 33.0084 29.9711 32.7547 30.0798C32.5011 30.1885 32.2268 30.2429 31.9318 30.2429ZM31.9318 29.8081C32.2579 29.8081 32.5451 29.7382 32.7936 29.5985C33.042 29.4587 33.2361 29.2647 33.3758 29.0162C33.5156 28.7626 33.5854 28.4728 33.5854 28.1467C33.5854 27.8155 33.5156 27.5256 33.3758 27.2772C33.2361 27.0287 33.042 26.8347 32.7936 26.6949C32.5451 26.55 32.2579 26.4775 31.9318 26.4775C31.6109 26.4775 31.3262 26.55 31.0778 26.6949C30.8294 26.8347 30.6353 27.0287 30.4955 27.2772C30.3558 27.5256 30.2859 27.8155 30.2859 28.1467C30.2859 28.4728 30.3558 28.7626 30.4955 29.0162C30.6353 29.2647 30.8294 29.4587 31.0778 29.5985C31.3262 29.7382 31.6109 29.8081 31.9318 29.8081ZM31.1477 29.2491V26.9822H31.9629C32.1958 26.9822 32.3873 27.0494 32.5374 27.184C32.6875 27.3134 32.7625 27.4868 32.7625 27.7042C32.7625 27.8491 32.7211 27.9785 32.6383 28.0924C32.5607 28.2062 32.4546 28.2916 32.32 28.3486L32.8557 29.2491H32.3976L31.9318 28.4262H31.5514V29.2491H31.1477ZM31.5514 28.0535H31.9784C32.0819 28.0535 32.1673 28.0225 32.2346 27.9604C32.307 27.8931 32.3433 27.8077 32.3433 27.7042C32.3433 27.5955 32.307 27.5101 32.2346 27.448C32.1621 27.3807 32.0664 27.3471 31.9473 27.3471H31.5514V28.0535Z" fill="#4FEAFF"/>
        </svg>
      </a>
      <span class="header-divider">|</span>
      <span class="header-github"><a href="https://github.com/revelata/deepkpi-agents" target="_blank" rel="noopener noreferrer">Benchmark Analysis (GitHub)</a></span>
    </div>
    <h1><span class="ticker">[TICKER]</span> &mdash; [Company Name]</h1>
    <div class="header-meta">
      <span><span class="meta-dot"></span>[Sector / business description]</span>
      <span>FY [date]</span>
      <span>CIK [##########]</span>
      <span>[N] benchmarks &middot; [M] segment groups</span>
    </div>
</div>

<!-- NOTE: Legend is NOT in the header. Place it after the grid-wrap closes. See "Grid skeleton" below. -->
```

## Document structure

```
HEADER (.header — single-column stack; no legend inside)
  .header-brand: <a for-ai-builders> + inlined SVG | Benchmark Analysis (GitHub)
  h1: <span class="ticker">[TICKER]</span> — [Company Name]
  .header-meta: sector (with .meta-dot) · FY · CIK · benchmark / segment counts

FINGERPRINT × COVERAGE GRID   ← hero element (DATA ONLY — see design principles + Rules)
  (first content after .header closes — no extra title bar above the grid)
  fp-grid table:
    group-header row: "Target" | "Group A label" | "Group B label (seg, purple)"
    col-header row: ticker pills (target=white, strong=green, partial=yellow, seg=purple)
    KPI rows:
      row-header: kpi-name + kpi-sub (left-aligned text)
      cells: target-col | match-strong | match-partial | match-none | seg-cell (centered numbers)
    NO prose / commentary rows inside the table (no "Why it matters", no fp-context, no thesis copy)
    group-header separator rows for new KPI groups (e.g. segment-specific KPIs)

LEGEND STRIP (.legend-strip — immediately AFTER </div><!-- grid-wrap -->)
  Horizontal row: Target | Strong match | Partial match | Segment (.legend-swatch squares)

BENCHMARK DETAIL
  section-label: "Benchmark Detail"
  cat-divider: "[Group name]"    ← plain text, font-weight:500, color:var(--text-dim)
  bmark-list:
    bmark-row: (bmark-id with ticker+name) + (bmark-data with kpi-comp blocks + bmark-note)
  cat-divider (segment, purple): "[Segment] — segment sub-benchmarks"
  bmark-list: (segment benchmarks)

KPI TRENDS OVER TIME
  section-label: "KPI trends over time"
  section-intro (one line under the label, optional but recommended):
    "Same metric, multiple companies, filing-resolved periods—compare **trajectory**, not just a single snapshot."
  chart-tabs: tabbed interface — one tab per fingerprint KPI (typically 2–3)
    Tab nav: .tab-btn elements, active tab highlighted in var(--cyan)
    Tab panels: one canvas per tab, Chart.js **line** chart (`type: 'line'`)
    X-axis: ordered reporting periods (FY… and/or FQ…), aligned across tickers where data exists
    Y-axis: KPI value (unit in tab label or chart subtitle); **log scale** when ranges are extreme (e.g. energy GWh)
    Y-axis QA (mandatory): **Double-check** the Y domain **includes every plotted value** for **every**
      series. Wrong `min`/`max`, `suggestedMin`/`suggestedMax`, or `beginAtZero` **clips** lines so they
      vanish—derive bounds from the **union** of all series or rely on Chart.js auto scales plus **padding**
      (`scales.y.grace`, e.g. 5–15%). Log scale: drop or skip non-positive values; confirm remaining lines render.
      If any legend entry has data but **no visible line**, treat it as a **blocking bug** and fix the axis.
    Datasets: **one line per company** (target + each benchmark). Line / point colors match ticker tier colors
      (same palette as the fingerprint grid — see **Chart color conventions**)
    DRI target segments: separate **lines** (e.g. dashed vs solid) within the target white/gray palette,
      not separate bars
    Data: multi-period pulls per the `kpi` and `benchmark` skills — this section showcases
      deepKPI **time series** competitors rarely get elsewhere; do not substitute a one-year grouped
      bar chart unless history is too sparse (&lt;3 shared periods); if forced, state the limitation in the chart caption

DIFF INSIGHT
  section-label: "Diff Insight"
  synthesis box (cyan left border): 2 paragraphs
    P1: what whole-company benchmarks reveal about the target
    P2: what the diff surfaced — segment gap or structural insight

NOTES
  section-label: "Notes"
  notes-list: ≤5 items (material caveats only — no scaling artifact explanations)

SOURCES
  section-label: "Sources"
  sources-table: Ticker | CIK | Coverage | FY end | Notes

FOOTER
  "Benchmark data sourced from SEC filings via Revelata deepKPI. All values link to source filing passages."
  Right: "FY[YEAR] · [Month YEAR]"

CLOSING CTA ("try it yourself")
  Same block as Analysis Pressure Test HTML: linked headline "Run this on any report with your agent:",
  two columns (Claude vs OpenClaw) with MCP URL, install.sh curl, and a final bullet (use the
  benchmark-specific prompts in the HTML skeleton below, not "Pressure test this report").

DISCLOSURES
  Same legal paragraphs as Analysis Pressure Test reports: Compensation, Third-Party Content,
  General, and the follow-on unlabeled paragraph (verbatim from pressure-test template).
```

---

## Key CSS classes

### Fingerprint grid

```css
.grid-wrap { overflow-x: auto; }
.fp-grid { border-collapse: collapse; width: 100%; font-size: 0.74rem; }

/* Sticky row label — KPI names only (left-aligned); grid has no prose rows */
.fp-grid th.row-header, .fp-grid td.row-header {
  position: sticky; left: 0; z-index: 2;
  background: var(--bg); min-width: 140px; max-width: 160px;
  text-align: left;
  vertical-align: top;
}

/* Group separator rows */
.fp-grid .group-header th {
  background: var(--bg); font-size: 0.56rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-dim);
  padding: 0.5rem 0.6rem 0.2rem; border-top: 1px solid var(--border);
}

/* Col header ticker pills */
.col-ticker { font-family: 'Figtree', monospace; font-weight: 700; font-size: 0.85rem; }
.col-ticker.target  { color: #fff; }
.col-ticker.match-strong { color: var(--green); }
.col-ticker.match-partial { color: var(--yellow); }
.col-ticker.seg { color: var(--purple); }

/* Data cells — tighter padding */
.fp-grid td.cell {
  padding: 0.35rem 0.55rem; border-bottom: 1px solid var(--border);
  text-align: center; vertical-align: middle; white-space: nowrap; min-width: 92px;
  font-family: 'JetBrains Mono', monospace; font-size: 0.71rem;
}

/* Row hover highlight */
.fp-grid tbody tr { transition: background 0.1s ease; }
.fp-grid tbody tr:not(.group-header):hover { background: rgba(255,255,255,0.035); }
.fp-grid tbody tr.group-header { background: var(--bg) !important; }

/* Cell states */
td.cell.target-col  { background: transparent; color: #fff; font-weight: 600; }
td.cell.match-strong { background: var(--green-mid); color: #86efac; }
td.cell.match-partial { background: var(--yellow-mid); color: #fde68a; }
td.cell.match-none   { background: transparent; color: var(--text-dim); }
td.cell.seg-cell     { background: var(--purple-dim); color: #d8b4fe; }

/* Delta sub-value */
.delta { display: block; font-size: 0.58rem; color: inherit; opacity: 0.65; margin-top: 0.08rem; }
```

### Benchmark cards

```css
.bmark-row {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 7px; display: grid; grid-template-columns: 100px 1fr; overflow: hidden;
}
.bmark-id {
  padding: 0.55rem 0.7rem; border-right: 1px solid var(--border);
  display: flex; flex-direction: column; justify-content: center;
  background: var(--surface2);
}
.bm-ticker { font-family: 'Figtree', monospace; font-weight: 700; font-size: 1rem; }
.bm-ticker.strong { color: var(--green); }
.bm-ticker.partial { color: var(--yellow); }
.bm-ticker.seg    { color: var(--purple); }

/* Category dividers — lighter than old tier labels */
.cat-divider {
  font-size: 0.6rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.07em;
  color: var(--text-dim); margin: 1rem 0 0.4rem; padding-bottom: 0.25rem;
  border-bottom: 1px solid var(--border);
}
/* Segment sub-benchmark group */
.cat-divider.seg { color: var(--purple); border-color: rgba(168,85,247,0.2); }
```

### Synthesis (Diff Insight)

```css
.synthesis {
  background: linear-gradient(135deg, rgba(79,234,255,0.04) 0%, transparent 60%);
  border: 1px solid rgba(79,234,255,0.15); border-left: 3px solid var(--cyan);
  border-radius: 7px; padding: 0.9rem 1.1rem; margin: 0.4rem 0 1.2rem;
}
.synthesis-title {
  font-size: 0.58rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.09em; color: var(--cyan); margin-bottom: 0.7rem;
}
```

### Page footer, closing CTA, and disclosures

Use the same rules as the `pressure-test` skill's `references/html-template.md`: data
footer, then **closing CTA** (self-serve install), then **disclosures**. Copy the CSS below
into the report’s `<style>` block (merge with existing rules).

```css
/* Footer */
.footer {
  margin-top: 2.5rem; padding-top: 1.4rem;
  border-top: 1px solid var(--border);
  font-size: 0.75rem; color: var(--text-dim); text-align: center;
}
.footer a { color: var(--cyan); text-decoration: none; }
.footer a:hover { text-decoration: underline; text-decoration-style: dotted; text-underline-offset: 3px; }

/* Disclosures */
.disclosures {
  margin-top: 1.5rem; padding-top: 1.2rem;
  border-top: 1px solid var(--border);
  font-size: 0.68rem; color: var(--text-dim);
  line-height: 1.6;
}
.disclosures p { margin-top: 0.7rem; }
.disclosures p:first-child { margin-top: 0; }
.disclosures strong { color: var(--text-muted); font-weight: 700; }

/* Closing CTA — plain on page background, between footer and disclosures */
.closing-cta {
  margin-top: 2rem;
  font-size: 0.82rem; color: var(--text-muted);
  line-height: 1.65;
}
.closing-cta .cta-setup-label {
  font-family: 'Figtree', 'Inter', system-ui, sans-serif;
  font-size: 1.15rem; font-weight: 700;
  margin-bottom: 1rem;
}
.closing-cta .cta-setup-label a {
  color: var(--cyan);
  text-decoration: none;
  border-bottom: 1px solid transparent;
  transition: border-color 0.15s ease;
}
.closing-cta .cta-setup-label a:hover {
  border-bottom-color: var(--cyan);
  text-decoration: none;
}
.closing-cta .cta-paths {
  display: grid; grid-template-columns: 1fr 1fr; gap: 1.6rem;
  margin-top: 0.3rem;
}
.closing-cta .cta-path-label {
  font-size: 0.7rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.07em;
  color: var(--text); margin-bottom: 0.5rem;
  padding-bottom: 0.4rem; border-bottom: 1px solid var(--border);
}
.closing-cta strong { color: var(--text-muted); font-weight: 700; }
.closing-cta a:not(.cta-setup-label *) {
  color: var(--cyan); text-decoration: underline;
  text-decoration-style: dotted; text-underline-offset: 2px;
}
.closing-cta a:not(.cta-setup-label *):hover { text-decoration-style: solid; }
.closing-cta ol { padding-left: 1.3rem; margin-top: 0.3rem; }
.closing-cta ol li { margin: 0.35rem 0; }
.closing-cta code {
  font-family: ui-monospace, 'SF Mono', Menlo, monospace;
  font-size: 0.75rem; color: var(--text);
  background: var(--surface); padding: 1px 6px; border-radius: 3px;
  border: 1px solid var(--border);
}

@media (max-width: 700px) {
  .closing-cta .cta-paths { grid-template-columns: 1fr; gap: 1.2rem; }
}
```

---

## Footer, “try it yourself” CTA, and disclosures (HTML)

Place **after** the main benchmark content (e.g. after Sources). The **closing CTA** is the same
self-serve (“try it yourself”) install block as in the `pressure-test` skill's `references/html-template.md`
(same headline link text, same steps 1–2). Step 3 must reference **benchmarking / peers / comps**
(see below), not pressure-testing an analyst report. The **disclosure** `<div class="disclosures">`
block matches the pressure-test template **word for word**.

```html
<!-- ===== FOOTER ===== -->
<div class="footer">
  Benchmark data sourced from SEC filings via
  <a href="https://www.revelata.com" target="_blank" rel="noopener noreferrer">Revelata deepKPI</a>.
  All values link to source filing passages.
</div>

<!-- ===== CLOSING CTA (try it yourself) ===== -->
<div class="closing-cta">
  <p class="cta-setup-label">
    <a href="https://github.com/revelata/deepkpi-agents" target="_blank" rel="noopener noreferrer">Try it yourself:</a>
  </p>
  <p>
    Installation and setup instructions live on GitHub:
    <a href="https://github.com/revelata/deepkpi-agents" target="_blank" rel="noopener noreferrer">deepkpi-agents</a>.
  </p>
</div>

<!-- ===== DISCLOSURES (same as Analysis Pressure Test template) ===== -->
<div class="disclosures">
  <p><strong>Compensation Disclosure:</strong> No part of the compensation of any Revelata personnel was, is, or will be directly or indirectly related to the specific views, conclusions, or recommendations expressed in this report, or to the coverage of any particular security or issuer herein.</p>

  <p><strong>Third-Party Content Disclosure:</strong> This report references and comments on research produced by a third-party analyst and their affiliated institution. Any such references are made for purposes of analysis and commentary only. The referenced analyst and their institution have not reviewed, approved, sponsored, or endorsed this report, Revelata, or any of Revelata's products, services, views, or conclusions, and no such endorsement or affiliation should be inferred. All commentary, opinions, and conclusions added by Revelata are solely those of Revelata and do not represent the views of the referenced analyst or their institution.</p>

  <p><strong>General Disclosure:</strong> The information contained in this equity research report is provided for informational purposes only and is not intended to constitute financial advice or a recommendation to buy, sell, or hold any securities. The opinions expressed herein are as of the date of publication and are subject to change without notice. This report does not take into account the investment objectives, financial situation, or specific needs of any individual investor.</p>

  <p>We are not a licensed financial advisor, broker, or dealer, and this report is not intended to provide investment, legal, accounting, or tax advice. Any reliance on the information contained in this report is at the sole discretion and risk of the user.</p>
</div>
```

---

## Grid skeleton

```html
<div class="grid-wrap">
<table class="fp-grid">
  <thead>
    <tr class="group-header">
      <th class="row-header"></th>
      <th colspan="1" style="color:rgba(255,255,255,0.35);">Target</th>
      <th colspan="N" style="color:var(--text-dim);">[Whole-company group label]</th>
      <!-- If segment sub-benchmarks exist: -->
      <th colspan="N" style="color:var(--purple);opacity:0.7;">[Segment] — sub-benchmarks</th>
    </tr>
    <tr class="col-header">
      <th class="row-header">KPI</th>
      <th style="text-align:center;"><div class="col-ticker target">[TARGET]</div></th>
      <th style="text-align:center;"><div class="col-ticker match-strong">[TICKER1]</div></th>
      <th style="text-align:center;"><div class="col-ticker match-partial">[TICKER2]</div></th>
      <th style="text-align:center;"><div class="col-ticker seg">[TICKER3]</div></th>
    </tr>
  </thead>
  <tbody>
    <!-- Optional group separator within body -->
    <tr class="group-header">
      <th class="row-header">[Group name e.g. "Whole-company"]</th>
      <th></th><th></th><th></th><th></th>
    </tr>
    <tr>
      <td class="row-header">
        <span class="kpi-name">[KPI label]</span>
        <span class="kpi-sub">[unit / period]</span>
      </td>
      <td class="cell target-col"><a href="provenance-url">[value]</a></td>
      <td class="cell match-strong"><a href="provenance-url">[value]</a><span class="delta">[±delta]</span></td>
      <td class="cell match-partial"><a href="provenance-url">[value]</a><span class="delta">[±delta]</span></td>
      <td class="cell match-none">—</td>
    </tr>
    <!-- For segment-specific KPI rows where all whole-company cols are blank: -->
    <tr class="group-header">
      <th class="row-header" style="color:var(--purple);opacity:0.7;">[Segment name]</th>
      <th></th><th></th><th></th><th></th>
    </tr>
    <tr>
      <td class="row-header"><span class="kpi-name">[Segment KPI]</span></td>
      <td class="cell target-col"><a href="provenance-url">[value]</a></td>
      <td class="cell match-none">—</td> <!-- all whole-company cols blank = gap visualization -->
      <td class="cell match-none">—</td>
      <td class="cell seg-cell"><a href="provenance-url">[value]</a></td>
    </tr>
  </tbody>
</table>
</div><!-- grid-wrap -->

<!-- ===== LEGEND STRIP (placed after grid, not in header) ===== -->
<div class="legend-strip">
  <div class="legend-item"><div class="legend-swatch" style="background:#fff;border:1px solid var(--border)"></div> Target</div>
  <div class="legend-item"><div class="legend-swatch" style="background:var(--green-mid)"></div> Strong match</div>
  <div class="legend-item"><div class="legend-swatch" style="background:var(--yellow-mid)"></div> Partial match</div>
  <div class="legend-item"><div class="legend-swatch" style="background:var(--purple-dim)"></div> Segment sub-benchmark</div>
</div>
```

---

## Benchmark card skeleton

```html
<div class="cat-divider">[Group label]</div>
<div class="bmark-list">
  <div class="bmark-row">
    <div class="bmark-id">
      <div class="bm-ticker strong">[TICKER]</div>  <!-- strong | partial | seg -->
      <div class="bm-name">[Full company name]</div>
    </div>
    <div class="bmark-data">
      <div class="kpi-comp">
        <div class="kc-label">[KPI name]</div>
        <div class="kc-val green"><a href="provenance-url">[value]</a></div>  <!-- green | yellow | dim -->
        <div class="kc-delta pos">vs [target value] ([±delta])</div>  <!-- pos | neg -->
      </div>
      <!-- More kpi-comp blocks -->
      <div class="bmark-note">[1-2 sentence analytical note]</div>
    </div>
  </div>
</div>

<!-- Segment sub-benchmark group -->
<div class="cat-divider" style="color:var(--purple);border-color:rgba(168,85,247,0.2);">
  [Segment name] — segment sub-benchmarks
</div>
```

---

## Chart color conventions

```js
// Line charts (KPI trends over time): use borderColor (+ borderWidth ~2) for each series;
// pointBackgroundColor / pointBorderColor match the same tier; fill: false (or a very light
// under-fill for the target line only—keep competitor lines unfilled for readability).
// Y scale: never hard-code min/max without verifying against ALL series; use grace/padding so
// lines are not clipped (invisible lines = invalid chart).
// Target: rgba(255,255,255,0.85) / rgba(255,255,255,1)
// Strong match: rgba(34,197,94,0.9) / rgba(34,197,94,1)
// Partial match: rgba(234,179,8,0.85) / rgba(234,179,8,1)
// Segment sub-benchmark: rgba(168,85,247,0.85) / rgba(168,85,247,1)
// Target segment breakdown (extra lines): rgba(255,255,255,0.45) dashed, or rgba(255,255,255,0.25) solid
// Negative values (if metric signed): rgba(239,68,68,0.85) / rgba(239,68,68,1)
```

---

## Rules

1. **Every number must be a hyperlink** to its `https://www.revelata.com/docviewer?pvid=NNNNN` URL.
2. **No tier badges** — color-coded tickers convey match quality; labels are plain category dividers.
3. **Target column = white**, not amber/yellow. Amber/yellow is reserved for partial benchmark match.
4. **Do not add scaling artifact footnotes** — silently correct values if confident they are wrong.
5. **Category dividers are plain text** (`font-weight: 500`, `color: var(--text-dim)`) not bold headers.
6. **Coverage gaps are visualized as blank rows** — rows where all benchmark columns show `—` make
   the gap immediately visible without prose explanation.
7. **Segment sub-benchmark headers and tickers use purple** — `var(--purple)` #a855f7.
8. **Chart colors match ticker colors** — same tier palette on **line** series as on grid headers (trends-over-time section).
9. **All segments or none** — if the target company has multiple reportable brand/business segments,
   include a grid row group for every segment (comps + AUV, or equivalent KPIs), not just the one
   with the best coverage. The benchmark columns for sparse segments will naturally show `—`, which
   is informative. Never arbitrarily cap the fingerprint at N dimensions or expose only the segments
   with the best benchmark matches.
10. **Page canvas**: use the **Page shell** `body` rules (`padding: 2.5rem`, `max-width: 900px`) and
    the single-column `.header` (no legend inside). For denser benchmark blocks below the legend strip,
    optional inner wrappers may use tighter padding; section label margins `1.5rem 0 0.4rem`; tab
    panels `0.7rem 0.9rem`; benchmark card data `0.45rem 0.7rem`.
11. **Footer + closing CTA + disclosures**: Always append the `.footer`, `.closing-cta`, and
    `.disclosures` blocks from the skeleton above — same legal copy as Analysis Pressure Test;
    same CTA headline and install steps 1–2; step 3 prompts benchmarking / peers / comps.
12. **Header logo:** Keep the **inlined SVG** from the **Page shell** section (same paths as
    the `pressure-test` skill's `references/html-template.md`). Never substitute an external logo
    image or a different wordmark.
13. **Fingerprint grid is data-only.** Do not add prose, thesis, "Why it matters", `tr.fp-context`,
    merged narrative cells, footnotes, or any non-KPI / non-numeric row inside `table.fp-grid`.
    Allowed: `thead` / `tbody` group-header rows, column headers, KPI label rows (`row-header` +
    `kpi-name` / `kpi-sub`), data cells (linked numbers, `—`, optional brief `match-none` note),
    and segment group separators. All interpretation belongs in **Benchmark Detail** cards and
    **Diff Insight**.
14. **Legend placement.** Render **`.legend-strip`** immediately **after** `</div><!-- grid-wrap -->`
    closes — never inside `.header`. Use the skeleton markup and `.legend-strip` CSS spacing
    (`margin-top: 0.6rem` gap above the strip).
15. **Header brand row.** In `.header-brand`, use only: inlined Revelata SVG (link to for-ai-builders),
    pipe divider, and **Benchmark Analysis (GitHub)**. Do not add a separate "deepKPI" product label
    in that row unless product marketing explicitly requests it.
16. **Time-series Y-axis must show the data.** Every KPI-over-time chart: verify the Y scale's range
    **contains all non-null points** for every dataset; add vertical padding (`grace`) when values cluster.
    Do not ship charts where lines disappear because the axis was restricted—recompute or remove bad bounds.
