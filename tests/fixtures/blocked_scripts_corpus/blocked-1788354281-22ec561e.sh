cd "$LOCALAPPDATA/hermes/scripts" && echo "=== result stores:" && ls results/ 2>/dev/null | head
echo "=== grep stored answers for provider-error text (real blast radius):"
grep -rlEi "usage limit has been reached|please try again later|api call failed after|HTTP 429|rate limit exceeded" results/ 2>/dev/null | head -20
echo "--- count of files with such text: $(grep -rlEi 'usage limit has been reached|please try again later|api call failed after|HTTP 429' results/ 2>/dev/null | wc -l)"
