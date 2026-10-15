from scripts.bac_cirt_monitor import parse_bac_filings, parse_fincen_actions, parse_occ_news


def test_parses_latest_bac_ir_filings_as_stable_primary_links():
    page = """
    <h2>Latest Current Report</h2><time datetime="2026-10-14T00:00:00">October 14, 2026</time>
    <h3>8-K</h3>
    <a href="/regulatory-and-other-filings/select-sec-filings/content/0001/bac-20261014.htm">View HTML</a>
    <h2>Latest Quarterly or Annual Filing</h2><time datetime="2026-07-31T00:00:00">July 31, 2026</time>
    <h3>10-Q</h3>
    <a href="/regulatory-and-other-filings/select-sec-filings/content/0002/bac-20260630.htm">View HTML</a>
    <h3>Q2 2026 Results</h3>
    <a href="/regulatory-and-other-filings/select-sec-filings/content/0003/q2-results.htm">Related material</a>
    """
    assert parse_bac_filings(page) == [
        {
            "source": "Bank of America IR / SEC filing",
            "published": "2026-10-14",
            "form": "8-K",
            "url": (
                "https://investor.bankofamerica.com/regulatory-and-other-filings/"
                "select-sec-filings/content/0001/bac-20261014.htm"
            ),
        },
        {
            "source": "Bank of America IR / SEC filing",
            "published": "2026-07-31",
            "form": "10-Q",
            "url": (
                "https://investor.bankofamerica.com/regulatory-and-other-filings/"
                "select-sec-filings/content/0002/bac-20260630.htm"
            ),
        },
    ]


def test_occ_feed_keeps_bank_specific_items_and_rejects_other_institutions():
    feed = """<rss><channel>
      <item><title>OCC Imposes Penalty Against Bank of America</title>
        <description>Deficient BSA compliance.</description><pubDate>13 Oct 2026</pubDate>
        <link>https://www.occ.gov/news-issuances/news-releases/2026/example.html</link></item>
      <item><title>OCC Updates General Guidance</title><description>All banks.</description>
        <link>https://www.occ.gov/news-issuances/news-releases/2026/other.html</link></item>
    </channel></rss>"""
    assert parse_occ_news(feed) == [{
        "source": "Office of the Comptroller of the Currency",
        "published": "13 Oct 2026",
        "title": "OCC Imposes Penalty Against Bank of America",
        "description": "Deficient BSA compliance.",
        "url": "https://www.occ.gov/news-issuances/news-releases/2026/example.html",
    }]


def test_fincen_action_list_tracks_bac_case_only():
    page = """
    <table><tr><th>Enforcement Action</th><th>Date</th><th>Matter</th><th>Institution</th></tr>
      <tr><td>
        <a href="/news/enforcement-actions/matter-bank-america-na">In the Matter of Bank of America, N.A.</a>
      </td>
        <td>10/13/2026</td><td>2026-03</td><td>Depository Institutions</td></tr>
      <tr><td><a href="/news/enforcement-actions/other">In the Matter of Another Bank</a></td>
        <td>10/14/2026</td><td>2026-04</td><td>Depository Institutions</td></tr>
    </table>
    """
    assert parse_fincen_actions(page) == [{
        "source": "FinCEN enforcement actions",
        "published": "10/13/2026",
        "matter": "2026-03",
        "title": "In the Matter of Bank of America, N.A.",
        "url": "https://www.fincen.gov/news/enforcement-actions/matter-bank-america-na",
    }]
