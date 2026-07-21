#!/usr/bin/env bash
# crawl.sh — stateless RSS/Atom crawler; emits a markdown digest to stdout. Deps: curl, xmlstarlet, pandoc.
set -euo pipefail
IFS=$'\n\t'

DEFAULT_UA='hermes-rss-feed/1.0 (+https://github.com/NousResearch/hermes-agent)'
LIMIT=10
UA="$DEFAULT_UA"
DEBUG=0
POSITIONAL=()

usage() {
    cat >&2 <<'EOF'
Usage: crawl.sh <feed-url-or-feeds-file> [--limit N] [--user-agent UA] [--debug]
  <feed-url-or-feeds-file>  http(s) URL, or path to a file with one URL per line ('#' comments ok)
  --limit N                 cap items per feed (default 10)
  --user-agent UA           override the HTTP User-Agent
  --debug                   set -x (verbose tracing)
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --limit)
            if [ "$#" -lt 2 ]; then echo "crawl.sh: --limit requires an argument" >&2; exit 2; fi
            LIMIT="$2"
            shift 2
            ;;
        --user-agent)
            if [ "$#" -lt 2 ]; then echo "crawl.sh: --user-agent requires an argument" >&2; exit 2; fi
            UA="$2"
            shift 2
            ;;
        --debug)
            DEBUG=1
            shift
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
            echo "crawl.sh: unknown flag: $1" >&2
            usage
            exit 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [ "$DEBUG" -eq 1 ]; then
    set -x
fi

if [ "${#POSITIONAL[@]}" -lt 1 ]; then
    usage
    exit 2
fi

TARGET="${POSITIONAL[0]}"

# Pre-flight: required tools.
missing=()
for cmd in curl xmlstarlet pandoc; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        missing+=("$cmd")
    fi
done
if [ "${#missing[@]}" -gt 0 ]; then
    os="$(uname -s 2>/dev/null || echo unknown)"
    case "$os" in
        Linux)  echo "crawl.sh: missing required tool(s): ${missing[*]} — try: sudo apt install xmlstarlet pandoc curl" >&2 ;;
        Darwin) echo "crawl.sh: missing required tool(s): ${missing[*]} — try: brew install xmlstarlet pandoc curl" >&2 ;;
        *)      echo "crawl.sh: missing required tool(s): ${missing[*]} — install xmlstarlet, pandoc, and curl" >&2 ;;
    esac
    exit 127
fi

# Temp file management.
TMPDIR_LOCAL="$(mktemp -d 2>/dev/null || mktemp -d -t 'crawl')"
cleanup() {
    if [ -n "${TMPDIR_LOCAL:-}" ] && [ -d "$TMPDIR_LOCAL" ]; then
        rm -rf "$TMPDIR_LOCAL"
    fi
}
trap cleanup EXIT INT TERM

# Build the list of feed URLs to crawl.
FEEDS=()
if [[ "$TARGET" == http://* || "$TARGET" == https://* ]]; then
    FEEDS+=("$TARGET")
else
    if [ ! -r "$TARGET" ]; then
        echo "crawl.sh: feeds file not readable: $TARGET" >&2
        exit 2
    fi
    while IFS= read -r line || [ -n "$line" ]; do
        # Strip CR (Windows line endings) and leading/trailing whitespace.
        line="${line%$'\r'}"
        # Trim leading whitespace.
        line="${line#"${line%%[![:space:]]*}"}"
        # Trim trailing whitespace.
        line="${line%"${line##*[![:space:]]}"}"
        if [ -z "$line" ]; then continue; fi
        case "$line" in
            \#*) continue ;;
        esac
        FEEDS+=("$line")
    done < "$TARGET"
fi

if [ "${#FEEDS[@]}" -eq 0 ]; then
    echo "crawl.sh: no feeds to crawl" >&2
    exit 0
fi

# Clean a piece of text: HTML -> plain via pandoc, strip CR, collapse whitespace, cap at 400 chars.
clean_text() {
    # Reads from stdin, writes one cleaned line to stdout.
    pandoc -f html -t plain --wrap=none 2>/dev/null \
        | tr -d '\r' \
        | tr '\n\t' '  ' \
        | tr -s ' ' \
        | sed -e 's/^ //' -e 's/ $//' \
        | head -c 400
}

# Process one feed URL. All output to stdout; warnings to stderr.
process_feed() {
    local url="$1"
    local body status http_code root title items_xpath feed_kind
    body="$TMPDIR_LOCAL/body.xml"

    # Fetch. Capture http code separately. --fail-with-body returns non-zero on >=400 but still gives the body.
    set +e
    http_code="$(curl --silent --location --compressed --max-time 30 \
        --user-agent "$UA" \
        --write-out '%{http_code}' \
        --output "$body" \
        "$url")"
    status="$?"
    set -e

    if [ "$status" -ne 0 ]; then
        echo "crawl.sh: curl exit $status for $url — skipping" >&2
        return 0
    fi
    case "$http_code" in
        2*|000) : ;;
        *)
            echo "crawl.sh: HTTP $http_code for $url — skipping" >&2
            return 0
            ;;
    esac

    if [ ! -s "$body" ]; then
        echo "crawl.sh: empty body for $url — skipping" >&2
        return 0
    fi

    # Detect root element.
    if ! root="$(xmlstarlet sel -t -v 'name(/*)' "$body" 2>/dev/null)"; then
        echo "crawl.sh: not valid XML: $url — skipping" >&2
        return 0
    fi

    case "$root" in
        rss)
            feed_kind=rss
            title="$(xmlstarlet sel -t -v '/rss/channel/title' "$body" 2>/dev/null || true)"
            items_xpath='/rss/channel/item'
            ;;
        feed)
            # Atom 1.0 uses xmlns="http://www.w3.org/2005/Atom" by default.
            # XPath 1.0 unqualified names only match the empty namespace, so we register a prefix.
            feed_kind=atom
            title="$(xmlstarlet sel -N a=http://www.w3.org/2005/Atom -t -v '/a:feed/a:title' "$body" 2>/dev/null || true)"
            items_xpath='/a:feed/a:entry'
            ;;
        *)
            echo "crawl.sh: unrecognized root element '$root' for $url — skipping" >&2
            return 0
            ;;
    esac

    if [ -z "$title" ]; then
        title="(untitled feed)"
    fi
    # Collapse whitespace in the feed title.
    title="$(printf '%s' "$title" | tr '\r\n\t' '   ' | tr -s ' ' | sed -e 's/^ //' -e 's/ $//')"

    printf '## %s — %s\n' "$title" "$url"

    # Extract items. Use tab as field delimiter, newline as record delimiter.
    # Fields: 1=title, 2=link, 3=date, 4=summary-html
    local items_tsv="$TMPDIR_LOCAL/items.tsv"
    if [ "$feed_kind" = rss ]; then
        # description first; if empty we'll still get one. content:encoded is in a namespace; xmlstarlet handles it via local-name.
        xmlstarlet sel -T -t \
            -m "$items_xpath" \
                -v 'normalize-space(title)' -o $'\t' \
                -v 'normalize-space(link)' -o $'\t' \
                -v 'normalize-space(pubDate)' -o $'\t' \
                -v "normalize-space((description | *[local-name()='encoded'])[1])" \
                -n \
            "$body" 2>/dev/null > "$items_tsv" || true
    else
        # Atom: prefer link[@rel='alternate']/@href, else first link/@href.
        # Summary: summary text, else content text.
        # All element names need the 'a:' prefix because of the default Atom namespace.
        xmlstarlet sel -N a=http://www.w3.org/2005/Atom -T -t \
            -m "$items_xpath" \
                -v 'normalize-space(a:title)' -o $'\t' \
                -v "normalize-space((a:link[@rel='alternate']/@href | a:link/@href)[1])" -o $'\t' \
                -v 'normalize-space((a:published | a:updated)[1])' -o $'\t' \
                -v 'normalize-space((a:summary | a:content)[1])' \
                -n \
            "$body" 2>/dev/null > "$items_tsv" || true
    fi

    if [ ! -s "$items_tsv" ]; then
        printf '(no items)\n\n'
        return 0
    fi

    # Apply per-feed limit.
    local limited="$TMPDIR_LOCAL/items.limited.tsv"
    head -n "$LIMIT" "$items_tsv" > "$limited"

    local any=0
    local item_title item_link item_date item_summary cleaned
    while IFS=$'\t' read -r item_title item_link item_date item_summary; do
        if [ -z "$item_title" ] && [ -z "$item_link" ]; then
            continue
        fi
        any=1
        [ -z "$item_title" ] && item_title="(untitled)"
        [ -z "$item_date" ]  && item_date="(no date)"

        if [ -n "$item_summary" ]; then
            cleaned="$(printf '%s' "$item_summary" | clean_text)"
        else
            cleaned=""
        fi

        printf -- '- **%s** — %s\n' "$item_title" "$item_date"
        if [ -n "$cleaned" ]; then
            printf '  %s\n' "$cleaned"
        fi
        if [ -n "$item_link" ]; then
            printf '  %s\n' "$item_link"
        fi
    done < "$limited"

    if [ "$any" -eq 0 ]; then
        printf '(no items)\n'
    fi
    printf '\n'
}

for feed_url in "${FEEDS[@]}"; do
    process_feed "$feed_url"
done
