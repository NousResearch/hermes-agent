#!/usr/bin/env bash
# pick.sh — interactive fzf picker over one or many RSS/Atom feeds. Deps: curl, xmlstarlet, pandoc, fzf (w3m optional).
set -euo pipefail
IFS=$'\n\t'

DEFAULT_UA='hermes-rss-feed/1.0 (+https://github.com/NousResearch/hermes-agent)'
LIMIT=25
UA="$DEFAULT_UA"
POSITIONAL=()

usage() {
    cat >&2 <<'EOF'
Usage: pick.sh <feed-url-or-feeds-file> [--limit N] [--user-agent UA]
  Interactive picker; uses fzf. Non-interactive contexts should use crawl.sh.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --limit)
            if [ "$#" -lt 2 ]; then echo "pick.sh: --limit requires an argument" >&2; exit 2; fi
            LIMIT="$2"
            shift 2
            ;;
        --user-agent)
            if [ "$#" -lt 2 ]; then echo "pick.sh: --user-agent requires an argument" >&2; exit 2; fi
            UA="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            while [ "$#" -gt 0 ]; do POSITIONAL+=("$1"); shift; done
            ;;
        -*)
            echo "pick.sh: unknown flag: $1" >&2
            usage
            exit 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [ "${#POSITIONAL[@]}" -lt 1 ]; then
    usage
    exit 2
fi

# TTY check — picker requires an interactive terminal.
if [ ! -t 1 ] || [ ! -t 0 ]; then
    echo "pick.sh: not a TTY — use scripts/crawl.sh for non-interactive contexts." >&2
    exit 0
fi

TARGET="${POSITIONAL[0]}"

# Pre-flight: required tools.
missing=()
for cmd in curl xmlstarlet pandoc fzf; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        missing+=("$cmd")
    fi
done
if [ "${#missing[@]}" -gt 0 ]; then
    os="$(uname -s 2>/dev/null || echo unknown)"
    case "$os" in
        Linux)  echo "pick.sh: missing required tool(s): ${missing[*]} — try: sudo apt install xmlstarlet pandoc curl fzf" >&2 ;;
        Darwin) echo "pick.sh: missing required tool(s): ${missing[*]} — try: brew install xmlstarlet pandoc curl fzf" >&2 ;;
        *)      echo "pick.sh: missing required tool(s): ${missing[*]} — install xmlstarlet, pandoc, curl, and fzf" >&2 ;;
    esac
    exit 127
fi

TMPDIR_LOCAL="$(mktemp -d 2>/dev/null || mktemp -d -t 'pick')"
cleanup() {
    if [ -n "${TMPDIR_LOCAL:-}" ] && [ -d "$TMPDIR_LOCAL" ]; then
        rm -rf "$TMPDIR_LOCAL"
    fi
}
trap cleanup EXIT INT TERM

# Build feed list.
FEEDS=()
if [[ "$TARGET" == http://* || "$TARGET" == https://* ]]; then
    FEEDS+=("$TARGET")
else
    if [ ! -r "$TARGET" ]; then
        echo "pick.sh: feeds file not readable: $TARGET" >&2
        exit 2
    fi
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%$'\r'}"
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        if [ -z "$line" ]; then continue; fi
        case "$line" in
            \#*) continue ;;
        esac
        FEEDS+=("$line")
    done < "$TARGET"
fi

if [ "${#FEEDS[@]}" -eq 0 ]; then
    echo "pick.sh: no feeds to fetch" >&2
    exit 0
fi

# Collect items from one feed into the merged TSV.
# Output fields per row: title <TAB> link <TAB> source <TAB> date
collect_feed() {
    local url="$1"
    local body http_code status root source items_xpath feed_kind
    body="$TMPDIR_LOCAL/body.xml"

    set +e
    http_code="$(curl --silent --location --compressed --max-time 30 \
        --user-agent "$UA" \
        --write-out '%{http_code}' \
        --output "$body" \
        "$url")"
    status="$?"
    set -e

    if [ "$status" -ne 0 ]; then
        echo "pick.sh: curl exit $status for $url — skipping" >&2
        return 0
    fi
    case "$http_code" in
        2*|000) : ;;
        *)
            echo "pick.sh: HTTP $http_code for $url — skipping" >&2
            return 0
            ;;
    esac

    if [ ! -s "$body" ]; then
        echo "pick.sh: empty body for $url — skipping" >&2
        return 0
    fi

    if ! root="$(xmlstarlet sel -t -v 'name(/*)' "$body" 2>/dev/null)"; then
        echo "pick.sh: not valid XML: $url — skipping" >&2
        return 0
    fi

    case "$root" in
        rss)
            feed_kind=rss
            source="$(xmlstarlet sel -t -v '/rss/channel/title' "$body" 2>/dev/null || true)"
            items_xpath='/rss/channel/item'
            ;;
        feed)
            # Atom 1.0: default namespace requires xmlstarlet -N prefix mapping.
            feed_kind=atom
            source="$(xmlstarlet sel -N a=http://www.w3.org/2005/Atom -t -v '/a:feed/a:title' "$body" 2>/dev/null || true)"
            items_xpath='/a:feed/a:entry'
            ;;
        *)
            echo "pick.sh: unrecognized root element '$root' for $url — skipping" >&2
            return 0
            ;;
    esac

    if [ -z "$source" ]; then
        source="$url"
    fi
    source="$(printf '%s' "$source" | tr '\r\n\t' '   ' | tr -s ' ' | sed -e 's/^ //' -e 's/ $//')"

    local raw="$TMPDIR_LOCAL/raw.tsv"
    if [ "$feed_kind" = rss ]; then
        xmlstarlet sel -T -t \
            -m "$items_xpath" \
                -v 'normalize-space(title)' -o $'\t' \
                -v 'normalize-space(link)' -o $'\t' \
                -v 'normalize-space(pubDate)' \
                -n \
            "$body" 2>/dev/null > "$raw" || true
    else
        xmlstarlet sel -N a=http://www.w3.org/2005/Atom -T -t \
            -m "$items_xpath" \
                -v 'normalize-space(a:title)' -o $'\t' \
                -v "normalize-space((a:link[@rel='alternate']/@href | a:link/@href)[1])" -o $'\t' \
                -v 'normalize-space((a:published | a:updated)[1])' \
                -n \
            "$body" 2>/dev/null > "$raw" || true
    fi

    if [ ! -s "$raw" ]; then
        return 0
    fi

    head -n "$LIMIT" "$raw" | while IFS=$'\t' read -r t l d; do
        if [ -z "$t" ] && [ -z "$l" ]; then continue; fi
        [ -z "$t" ] && t="(untitled)"
        [ -z "$d" ] && d="(no date)"
        printf '%s\t%s\t%s\t%s\n' "$t" "$l" "$source" "$d"
    done
}

ALL="$TMPDIR_LOCAL/all.tsv"
: > "$ALL"
for feed_url in "${FEEDS[@]}"; do
    collect_feed "$feed_url" >> "$ALL"
done

if [ ! -s "$ALL" ]; then
    echo "pick.sh: no items collected" >&2
    exit 0
fi

# fzf: show title (col 1) + date (col 4); preview shows the link (col 2).
selection="$(fzf --delimiter=$'\t' --with-nth=1,4 --preview 'echo {2}' --prompt='item> ' < "$ALL" || true)"
if [ -z "$selection" ]; then
    exit 0
fi

# Extract the link (field 2).
url="$(printf '%s' "$selection" | awk -F '\t' '{print $2}')"
if [ -z "$url" ]; then
    echo "pick.sh: selected item has no link" >&2
    exit 0
fi

printf '%s\n' "$url"

# Render the article: prefer curl + pandoc -> less; fall back to w3m; else just print the URL.
page_output() {
    if command -v less >/dev/null 2>&1; then
        less -R
    else
        cat
    fi
}

render_with_pandoc() {
    local html
    html="$TMPDIR_LOCAL/article.html"
    if ! curl --silent --location --compressed --max-time 30 --user-agent "$UA" --output "$html" "$url"; then
        return 1
    fi
    if [ ! -s "$html" ]; then
        return 1
    fi
    if ! pandoc -f html -t plain --wrap=none "$html" 2>/dev/null | page_output; then
        return 1
    fi
    return 0
}

if ! render_with_pandoc; then
    if command -v w3m >/dev/null 2>&1; then
        w3m -dump "$url" | page_output || printf '%s\n' "$url"
    else
        printf '%s\n' "$url"
    fi
fi
