#!/usr/bin/env python3
"""Full-git-history secret and identity scan. Stdlib only, no dependencies.

Usage:
    history_scan.py [--personal FILE] [PATH ...]

Each PATH is a git repo (bare/mirror or normal) or a directory containing
<name>.git mirror clones. With no PATH: scans the cwd if it is a repo,
otherwise every *.git directory inside it.

Scans every blob reachable from ANY ref (git rev-list --objects --all),
binaries included, plus commit messages and annotated tag messages, and
reports every author/committer identity in history.

Personal patterns: one regex per line (blank lines and # comments ignored)
from --personal FILE, or ~/.hermes/personal-patterns.txt if it exists.
The current user's username and home directory path are always checked.
"""
import subprocess, re, sys, os, collections, threading

SECRET_PATTERNS = [
    ("stripe-style sk_", rb"sk_[A-Za-z0-9]{16,}"),
    ("google-api-key", rb"AIza[0-9A-Za-z_\-]{35}"),
    ("google-oauth-secret", rb"GOCSPX-[A-Za-z0-9_\-]{20,}"),
    ("github-token", rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    ("aws-key", rb"AKIA[0-9A-Z]{16}"),
    ("openai-style sk-", rb"sk-[A-Za-z0-9_\-]{20,}"),
    ("xai-key", rb"xai-[A-Za-z0-9]{20,}"),
    ("huggingface-token", rb"hf_[A-Za-z0-9]{30,}"),
    ("npm-token", rb"npm_[A-Za-z0-9]{36}"),
    ("agentmail-key", rb"\bam_[A-Za-z0-9]{20,}"),
    ("telegram-bot-token", rb"\b[0-9]{8,10}:AA[A-Za-z0-9_\-]{33}\b"),
    ("slack-token", rb"xox[baprs]-[A-Za-z0-9\-]{10,}"),
    ("slack-webhook", rb"hooks\.slack\.com/services/T[A-Za-z0-9/]+"),
    ("sendgrid-key", rb"SG\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}"),
    ("private-key-block", rb"-----BEGIN [A-Z ]*PRIVATE KEY"),
    ("jwt", rb"eyJ[A-Za-z0-9_\-]{15,}\.eyJ[A-Za-z0-9_\-]{15,}"),
    ("bearer-literal", rb"[Bb]earer\s+[A-Za-z0-9_\-\.=]{20,}"),
    ("generic-assignment", rb"(?i)(api[_\-]?key|apikey|api_secret|client_secret|access_token|auth_token|passwd|password)[\"']?\s*[:=]\s*[\"'][^\"'\s]{8,}[\"']"),
    # bare .env-style lines, any case: API_KEY=value, db_password=value, TOKEN=value.
    # The key word must stand alone or be joined by underscores (so keyUsage and
    # MAX_TOKENS don't match); the value must not look like a call.
    ("env-assignment", rb"(?im)^(?:export[ \t]+)?(?:[A-Z0-9_]*_)?(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)(?:_[A-Z0-9_]*)?[ \t]*=[ \t]*[^\s\"'#$(][^\s\"'()]{7,}(?=[\s\"']|$)"),
    ("solana-keypair-json", rb"\[(?:\s*\d{1,3}\s*,){63}\s*\d{1,3}\s*\]"),
]
EMAIL = re.compile(rb"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
EMAIL_NOISE = re.compile(rb"(?i)(example\.com|users\.noreply\.github\.com|@2x|sentry|schema|\.png|\.jpg|@[0-9]+\.[0-9]+|node_modules|@babel|@types|@keyframes|@media)")

def load_personal_patterns(argv):
    """Return ([(label, compiled_bytes_pattern)], remaining_argv)."""
    path = None
    if "--personal" in argv:
        i = argv.index("--personal")
        try:
            path = argv[i + 1]
        except IndexError:
            sys.exit("--personal needs a file argument")
        argv = argv[:i] + argv[i + 2:]
    else:
        default = os.path.join(
            os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")),
            "personal-patterns.txt")
        if os.path.isfile(default):
            path = default

    pats = []
    home = os.path.expanduser("~")
    if home and home not in ("/", "\\"):
        pats.append(("home-path", re.compile(re.escape(home.encode()))))
        user = os.path.basename(home.rstrip("/\\"))
        if len(user) >= 3:
            pats.append(("username", re.compile(rb"(?i)\b" + re.escape(user.encode()) + rb"\b")))
    if path:
        with open(path, "rb") as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith(b"#"):
                    continue
                try:
                    pats.append((f"personal:{n}", re.compile(line)))
                except re.error as e:
                    print(f"!! bad personal pattern line {n}: {e}", file=sys.stderr)
    return pats, argv

def resolve_repos(paths):
    paths = paths or ["."]
    repos = []
    for p in paths:
        p = os.path.abspath(p)
        if subprocess.run(["git", "-C", p, "rev-parse", "--git-dir"],
                          capture_output=True).returncode == 0:
            repos.append(p)
        elif os.path.isdir(p):
            sub = [os.path.join(p, d) for d in sorted(os.listdir(p)) if d.endswith(".git")]
            if not sub:
                print(f"!! {p}: not a repo and holds no *.git dirs", file=sys.stderr)
            repos.extend(sub)
        else:
            print(f"!! {p}: no such directory", file=sys.stderr)
    return repos

def git(repo, *args, binary=False):
    r = subprocess.run(["git", "-C", repo] + list(args), capture_output=True)
    return r.stdout if binary else r.stdout.decode(errors="replace")

def cat_blobs(repo, shas):
    """Yield (sha, data) for each blob from one `cat-file --batch` process."""
    p = subprocess.Popen(["git", "-C", repo, "cat-file", "--batch"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    def feed():
        for sha in shas:
            p.stdin.write(sha.encode() + b"\n")
        p.stdin.close()
    threading.Thread(target=feed, daemon=True).start()
    while True:
        header = p.stdout.readline()
        if not header:
            break
        parts = header.split()
        if len(parts) < 3:  # "<sha> missing"
            continue
        data = p.stdout.read(int(parts[2]))
        p.stdout.read(1)  # newline after the object body
        yield parts[0].decode(), data
    p.wait()

def blob_paths(repo, shas):
    """{sha: every path that ever held it} for the given blobs, from one pass
    over the history's raw diffs (rev-list --objects names each blob once)."""
    out = git(repo, "log", "--all", "--raw", "--no-renames", "--no-abbrev",
              "--format=", "-z", binary=True)
    paths = collections.defaultdict(set)
    fields = out.split(b"\x00")  # ":<mode> <mode> <old> <new> <status>", "<path>", ...
    for meta, path in zip(fields[::2], fields[1::2]):
        parts = meta.split()
        if len(parts) == 5 and parts[4] != b"D":
            sha = parts[3].decode()
            if sha in shas:
                paths[sha].add(path.decode(errors="replace"))
    return paths

def main():
    personal, argv = load_personal_patterns(sys.argv[1:])
    all_patterns = SECRET_PATTERNS + [(n, p.pattern) for n, p in personal]

    def scan_bytes(data, hits, where):
        found = False
        for name, pat in all_patterns:
            for m in re.finditer(pat, data):
                frag = m.group(0)[:70]
                hits[(name, where)].add(frag.decode(errors="replace"))
                found = True
        return found

    exit_hits = 0
    for repo in resolve_repos(argv):
        print("#" * 70)
        print("## REPO:", repo)
        ids = git(repo, "log", "--all", "--format=%an <%ae> | %cn <%ce>")
        counts = collections.Counter(ids.strip().splitlines())
        print("-- commit identities:")
        for k, v in counts.most_common():
            print(f"   {v:4d}x {k}")

        hits = collections.defaultdict(set)
        email_hits = collections.defaultdict(set)

        # -z separates commits with NUL; each record is "<hash>\n<body>".
        msgs = git(repo, "log", "--all", "-z", "--format=%H%n%B", binary=True)
        for rec in msgs.split(b"\x00"):
            sha, _, body = rec.partition(b"\n")
            if sha and body.strip():
                scan_bytes(body, hits, f"commit-msg {sha[:10].decode()}")
        tags = git(repo, "tag", "-l", "--format=%(objectname) %(contents)", binary=True)
        if tags.strip():
            scan_bytes(tags, hits, "tag-messages")

        out = git(repo, "rev-list", "--objects", "--all")
        blobs = {}
        for line in out.splitlines():
            parts = line.split(" ", 1)
            if len(parts) == 2 and parts[1]:
                blobs.setdefault(parts[0], parts[1])
        types = {}
        p = subprocess.run(["git", "-C", repo, "cat-file", "--batch-check"],
                           input="\n".join(blobs.keys()).encode(), capture_output=True)
        for line in p.stdout.decode().splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[1] == "blob":
                types[parts[0]] = int(parts[2])

        wanted = []
        for sha, path in blobs.items():
            if sha not in types:
                continue
            if types[sha] > 20_000_000:
                print(f"   !! skipped huge blob {path} ({types[sha]}b)")
                continue
            wanted.append(sha)

        nbin = 0
        hit_blobs = collections.defaultdict(set)
        for sha, data in cat_blobs(repo, wanted):
            path = blobs[sha]
            is_bin = b"\x00" in data[:8192]
            if is_bin:
                nbin += 1
            if scan_bytes(data, hits, path):
                hit_blobs[path].add(sha)
            if not is_bin:
                for m in EMAIL.finditer(data):
                    e = m.group(0)
                    if not EMAIL_NOISE.search(e):
                        email_hits[e.decode(errors="replace")].add(path)

        # The same content may live at other paths too; resolve those for hits only.
        also = {}
        if hit_blobs:
            found = blob_paths(repo, set().union(*hit_blobs.values()))
            for path, shas in hit_blobs.items():
                more = set().union(*(found.get(sha, set()) for sha in shas)) - {path}
                if more:
                    also[path] = sorted(more)

        print(f"-- scanned {len(types)} blobs ({nbin} binary) + commit/tag messages")
        if hits:
            print("-- pattern hits:")
            for (name, where), frags in sorted(hits.items()):
                loc = where
                if where in also:
                    loc += f" (also at {', '.join(also[where][:4])})"
                for f in sorted(frags)[:6]:
                    print(f"   [{name}] {loc}: {f}")
        else:
            print("-- pattern hits: none")
        if email_hits:
            print("-- emails found in content:")
            for e, paths in sorted(email_hits.items()):
                print(f"   {e}  ({', '.join(sorted(paths)[:4])})")
        n = sum(len(v) for v in hits.values())
        exit_hits += n
        print(f"RESULT: {n} pattern hit(s), {len(email_hits)} distinct content email(s)")
        print()
    # Nonzero exit when anything hit, so callers can gate on it. Hits still
    # need human judgment (placeholders are fine) — this is a flag, not a verdict.
    sys.exit(1 if exit_hits else 0)

if __name__ == "__main__":
    main()
