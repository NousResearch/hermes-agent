# HTML Template Reference

This is the canonical HTML structure for every Analysis Pressure Test report.
Copy this skeleton and fill in the dynamic content. Do not deviate from the
CSS custom properties, class names, or layout structure — they are the Revelata
brand identity for this product.

**Logo asset:** The Revelata wordmark is **inlined as SVG** directly in the
header. No external asset is referenced and no file needs to be copied next
to the HTML. The logo links to `https://www.revelata.com/for-ai-builders`.

## Full HTML skeleton

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>[TICKER] Analysis Pressure Test | Revelata</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Figtree:wght@400;500;600;700&display=swap');
  :root {
    --bg: #0e1117; --surface: #161b22; --surface2: #1c2129;
    --border: #2a313a; --border-light: #353d47;
    --text: #e6edf3; --text-muted: #8b949e; --text-dim: #6e7681;
    --cyan: #4FEAFF; --cyan-dim: rgba(79,234,255,0.15); --cyan-glow: rgba(79,234,255,0.08);
    --blue: #90caf9; --blue-dim: rgba(144,202,249,0.12);
    --amber: #f59e0b; --amber-dim: rgba(245,158,11,0.12); --amber-text: #fbbf24;
    --red: #f87171; --red-dim: rgba(248,113,113,0.10);
    --green: #4ade80; --green-dim: rgba(74,222,128,0.10);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Inter', system-ui, sans-serif;
    background: var(--bg); color: var(--text);
    line-height: 1.7; padding: 2.5rem;
    max-width: 900px; margin: 0 auto;
    font-size: 14.5px; -webkit-font-smoothing: antialiased;
  }

  /* Header */
  .header { margin-bottom: 2.5rem; }
  .header-brand { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.8rem; flex-wrap: wrap; }
  .header-brand .header-logo {
    display: block; height: 28px; width: auto; flex-shrink: 0;
  }
  .header-brand span {
    font-family: 'Figtree', 'Inter', system-ui, sans-serif;
    font-size: 1.1rem; font-weight: 600;
    color: var(--cyan);
  }
  .header-brand span a { color: var(--cyan); font-family: inherit; }
  .header-brand .header-divider {
    color: var(--border-light);
    font-weight: 300;
    font-size: 1.5rem;
    line-height: 1;
    margin: 0 0.1rem;
  }
  .header a { color: inherit; text-decoration: none; }
  .header a:hover { text-decoration: underline; text-decoration-style: dotted; text-underline-offset: 3px; }
  h1 {
    font-size: 1.6rem; font-weight: 700;
    letter-spacing: -0.03em; color: #fff; margin-bottom: 0.3rem;
  }
  .subtitle { color: var(--text-muted); font-size: 0.85rem; margin-bottom: 1.5rem; }
  .subtitle strong { color: var(--red); font-weight: 600; }

  /* Legend */
  .legend {
    display: flex; gap: 2rem; padding: 0.9rem 1.2rem;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; font-size: 0.8rem; color: var(--text-muted);
  }
  .legend-item { display: flex; align-items: center; gap: 0.5rem; }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }

  /* Price since report */
  .pricebox {
    margin-top: 1rem;
    background: var(--surface); border: 1px solid var(--border);
    border-left: 3px solid var(--cyan); border-radius: 8px;
    padding: 0.9rem 1.2rem;
  }
  .pricebox-title {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.07em; color: var(--cyan);
    margin-bottom: 0.35rem;
  }
  .pricebox-body { font-size: 0.85rem; color: var(--text-muted); }
  .pricebox-body strong { color: #fff; font-weight: 650; }
  .pricebox-note { margin-top: 0.35rem; font-size: 0.75rem; color: var(--text-dim); }

  /* Section headings */
  h2 {
    font-size: 1.1rem; font-weight: 700; color: #fff;
    margin: 3rem 0 0.4rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
  }
  h2 .num { color: var(--cyan); margin-right: 0.4rem; }

  /* Analyst claim banner */
  .claim {
    background: var(--surface); border: 1px solid var(--border);
    border-left: 3px solid var(--amber); border-radius: 6px;
    padding: 0.9rem 1.1rem; margin: 1rem 0 1.2rem;
    font-size: 0.88rem; color: var(--text-muted);
  }
  .claim strong { color: var(--amber-text); font-weight: 600; }
  .claim .src {
    font-size: 0.7rem; font-weight: 600; color: var(--amber);
    text-transform: uppercase; letter-spacing: 0.06em; margin-right: 0.3rem;
  }

  /* Evidence cards */
  .evidence { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0 1.5rem; }
  .ev-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 1.1rem 1.2rem;
    font-size: 0.85rem; line-height: 1.65; color: var(--text-muted);
  }
  .ev-card p { margin-bottom: 0.6rem; }
  .ev-card p:last-child { margin-bottom: 0; }
  .ev-support { border-top: 3px solid var(--green); }
  .ev-counter { border-top: 3px solid var(--red); }
  .ev-label {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.07em; margin-bottom: 0.7rem;
    display: flex; align-items: center; gap: 0.4rem;
  }
  .ev-support .ev-label { color: var(--green); }
  .ev-counter .ev-label { color: var(--red); }
  .ev-label svg { width: 13px; height: 13px; }

  /* Inline data highlights */
  .d { font-weight: 600; }
  .d-r { color: var(--amber-text); }
  .d-s { color: var(--cyan); }
  .d-s a, .ev-card a, .synthesis a:not(.footer a) {
    color: var(--cyan); text-decoration: underline;
    text-decoration-style: dotted; text-underline-offset: 2px;
  }
  .d-s a:hover, .ev-card a:hover, .synthesis a:hover {
    text-decoration-style: solid;
  }

  /* Source pills */
  .src-pill {
    display: inline-block; font-size: 0.6rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.05em;
    padding: 1px 5px; border-radius: 3px;
    vertical-align: middle; margin-right: 2px;
  }
  .src-r { background: var(--amber-dim); color: var(--amber); }
  .src-s { background: var(--cyan-dim); color: var(--cyan); }

  /* Charts */
  .chart-card {
    background: var(--surface); border: 1px solid var(--border);
    border-left: 3px solid var(--cyan); border-radius: 10px;
    padding: 1.3rem 1.4rem; margin: 1.2rem 0;
  }
  .chart-header {
    display: flex; justify-content: space-between;
    align-items: baseline; margin-bottom: 0.8rem;
  }
  .chart-title { font-size: 0.8rem; font-weight: 600; color: var(--text); letter-spacing: 0.01em; }
  .chart-src { font-size: 0.65rem; color: var(--text-dim); }
  canvas { max-height: 280px; }

  /* Synthesis */
  .synthesis {
    background: var(--cyan-glow);
    border: 1px solid rgba(79,234,255,0.25);
    border-radius: 10px; padding: 1.6rem 1.8rem; margin: 3rem 0 1rem;
  }
  .synthesis h2 { border: none; margin-top: 0; color: var(--cyan); }
  .synthesis p { margin-bottom: 1rem; color: var(--text-muted); font-size: 0.9rem; }
  .synthesis p:last-child { margin-bottom: 0; }

  /* Footer */
  .footer {
    margin-top: 2.5rem; padding-top: 1.4rem;
    border-top: 1px solid var(--border);
    font-size: 0.75rem; color: var(--text-dim); text-align: center;
  }
  .footer a { color: var(--cyan); text-decoration: none; }
  .footer a:hover { text-decoration: underline; text-decoration-style: dotted; }

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

  /* Responsive */
  @media (max-width: 700px) {
    body { padding: 1.2rem; }
    .evidence { grid-template-columns: 1fr; }
    .legend { flex-direction: column; gap: 0.5rem; }
    .closing-cta .cta-paths { grid-template-columns: 1fr; gap: 1.2rem; }
  }
</style>
</head>
<body>

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
    <span><a href="https://github.com/revelata/deepkpi-agents" target="_blank" rel="noopener noreferrer">Analysis Pressure Test (GitHub)</a></span>
  </div>
  <h1>[Company] ([TICKER]): Pressure-Testing the [Firm] [Action]</h1>
  <p class="subtitle">
    [Firm] [Division] &mdash; "[Report Title]" &mdash; [Date] &mdash;
    Rating: [Old] &rarr; <strong>[New]</strong>
  </p>

  <div class="legend">
    <div class="legend-item">
      <div class="legend-dot" style="background:var(--amber)"></div>
      Analyst report data
    </div>
    <div class="legend-item">
      <div class="legend-dot" style="background:var(--cyan)"></div>
      Supplemental data from SEC filings (deepKPI)
    </div>
  </div>

  <!-- Price reaction context (fill these placeholders) -->
  <div class="pricebox">
    <div class="pricebox-title">Price since report</div>
    <div class="pricebox-body">
      Report date close: <strong>$[P_REPORT]</strong> &nbsp;·&nbsp;
      Most recent close: <strong>$[P_NOW]</strong> &nbsp;·&nbsp;
      Move: <strong>[DELTA_PCT]%</strong>
    </div>
    <div class="pricebox-note">
      If the report is &gt;3 weeks old, add 2–4 sentences on what moved and which SEC KPIs
      would have hinted at it. If &gt;3 months old, contrast data available then vs new filings since.
    </div>
  </div>
</div>

<!-- ===== ARGUMENT N ===== -->
<!-- Repeat this block for each argument (4-6 total) -->

<h2><span class="num">01</span> [Argument Title]</h2>

<div class="claim">
  <span class="src">[FIRM] says:</span>
  [1-2 sentence summary of the analyst's claim. Bold the key numbers.]
</div>

<!-- Optional: 1-2 chart cards if visual comparison helps -->
<div class="chart-card">
  <div class="chart-header">
    <span class="chart-title">[Chart Title]</span>
    <span class="chart-src">
      <span class="src-pill src-s">SEC</span> [Filing type]
    </span>
  </div>
  <canvas id="c[N]"></canvas>
</div>

<div class="evidence">
  <div class="ev-card ev-support">
    <div class="ev-label">
      <svg viewBox="0 0 16 16" fill="currentColor">
        <path d="M8 1a7 7 0 100 14A7 7 0 008 1zm1 10H7V7h2v4zm0-5H7V4h2v2z"/>
      </svg>
      Supports the thesis
    </div>
    <p>
      <span class="src-pill src-s">SEC</span>
      [Analysis with hyperlinked numbers:
       <span class="d d-s"><a href="provenance-url" target="_blank">$XXM</a></span>]
    </p>
  </div>
  <div class="ev-card ev-counter">
    <div class="ev-label">
      <svg viewBox="0 0 16 16" fill="currentColor">
        <path d="M8 1a7 7 0 100 14A7 7 0 008 1zm1 10H7V7h2v4zm0-5H7V4h2v2z"/>
      </svg>
      Complicates the thesis
    </div>
    <p>
      <span class="src-pill src-s">SEC</span>
      [Counter-analysis with hyperlinked numbers]
    </p>
    <p>
      <span class="src-pill src-r">[FIRM]</span>
      [Relevant point from the report itself that complicates the thesis]
    </p>
  </div>
</div>

<!-- ===== END OF ARGUMENT BLOCKS ===== -->

<!-- ===== SYNTHESIS ===== -->
<div class="synthesis">
  <h2>The Deeper Picture</h2>
  <p>[Paragraph 1: what the analyst gets right, with hyperlinked numbers]</p>
  <p>[Paragraph 2: what the analyst misses, the key risk/decision point]</p>
</div>

<!-- ===== FOOTER ===== -->
<div class="footer">
  Supplemental data sourced from SEC filings via
  <a href="https://www.revelata.com" target="_blank">Revelata deepKPI</a>.
</div>

<!-- ===== CLOSING CTA ===== -->
<!-- Plain on page background (no box). Keep this CTA minimal: link out to GitHub
     for installation and setup (do not embed shell commands here). -->
<div class="closing-cta">
  <p class="cta-setup-label">
    <a href="https://github.com/revelata/deepkpi-agents" target="_blank" rel="noopener noreferrer">Run this on any report with your agent:</a>
  </p>
  <p>
    Install and setup instructions (Claude connector + OpenClaw) live on GitHub:
    <a href="https://github.com/revelata/deepkpi-agents" target="_blank" rel="noopener noreferrer">deepkpi-agents</a>.
  </p>
</div>

<!-- ===== DISCLOSURES ===== -->
<div class="disclosures">
  <p><strong>Compensation Disclosure:</strong> No part of the compensation of any Revelata personnel was, is, or will be directly or indirectly related to the specific views, conclusions, or recommendations expressed in this report, or to the coverage of any particular security or issuer herein.</p>

  <p><strong>Third-Party Content Disclosure:</strong> This report references and comments on research produced by a third-party analyst and their affiliated institution. Any such references are made for purposes of analysis and commentary only. The referenced analyst and their institution have not reviewed, approved, sponsored, or endorsed this report, Revelata, or any of Revelata's products, services, views, or conclusions, and no such endorsement or affiliation should be inferred. All commentary, opinions, and conclusions added by Revelata are solely those of Revelata and do not represent the views of the referenced analyst or their institution.</p>

  <p><strong>General Disclosure:</strong> The information contained in this equity research report is provided for informational purposes only and is not intended to constitute financial advice or a recommendation to buy, sell, or hold any securities. The opinions expressed herein are as of the date of publication and are subject to change without notice. This report does not take into account the investment objectives, financial situation, or specific needs of any individual investor.</p>

  <p>We are not a licensed financial advisor, broker, or dealer, and this report is not intended to provide investment, legal, accounting, or tax advice. Any reliance on the information contained in this report is at the sole discretion and risk of the user.</p>
</div>

<!-- ===== CHARTS ===== -->
<script>
// Chart.js global defaults for dark theme
Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#2a313a';
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.font.size = 11;

// Blue/cyan palette — ALL SEC data charts use only these colors
const B1 = '#4FEAFF';  // cyan — primary
const B2 = '#90caf9';  // light blue — secondary
const B3 = '#5b8def';  // mid blue — tertiary
const B4 = '#a78bfa';  // periwinkle — quaternary
const grid = { color: 'rgba(255,255,255,0.05)' };

// [Insert chart constructors here — see chart-patterns.md for examples]
</script>
</body>
</html>
```

## Key rules

1. **Never change the CSS custom properties** — they are the brand identity
2. **Logo is inlined as SVG in the header** — no external file reference. Links to `https://www.revelata.com/for-ai-builders`.
3. **Header divider is a vertical pipe** (`|`) styled in `--border-light` at 1.5rem — sits between the wordmark and the "Analysis Pressure Test (GitHub)" label.
4. **Closing CTA is minimal** — do not embed install commands in the HTML. Link to the GitHub repo for setup.

