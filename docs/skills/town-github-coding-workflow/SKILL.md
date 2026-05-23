## GITHUB CODING WORKFLOW

When accessing any file in a GitHub repository — including reading a plan document, exploring the structure, editing code, or implementing features — use the VFS-based workflow below. This applies even for reading a single file. **Never keep large file contents in the context window** — read into VFS, work in VFS, commit from VFS.

> **Never use `github_get_file` as your first approach** — it truncates files over 15KB and puts content in the context window. Always copy to VFS first, then read/query from VFS.
>
> For any GitHub file (even a single file like a plan document):
> ```
> town_cp source: "github://owner/repo/path/file.md?ref=main" dest: "vfs:///session/files/file.md"
> town_read uri: "vfs:///session/files/file.md" limit: 200   # use limit if sizeBytes > 25000
> ```

### Step-by-step

**1. Pull the repo or subdirectory to VFS:**

```
town_cp source: "github://owner/repo/src?ref=main" dest: "vfs:///session/files/project/src"
```

- Check that `filesCopied > 0` in the result. If it is 0 or you get a "truncated tree" error or "Failed to copy any ... files" error, copy a specific subdirectory path instead (e.g., `/src`, `/convex`). Use `town_ls /github/owner/repo` first to see the structure.
- You can copy multiple subdirectories with separate `town_cp` calls.

**2. Read files from VFS:**

```
town_read uri: "vfs:///session/files/project/src/App.tsx"
```

- Do NOT use `github_get_file` — always use `town_read` from VFS instead. If you haven't copied the file to VFS yet, do that first with `town_cp`.
- `town_cp` returns `copiedFiles[*].sizeBytes` for every file — use this to identify which files are over 25000 bytes and will need `limit: 200` before you start reading.
- For large files (sizeBytes > 25000): use `limit: 200` and paginate: `{offset: 1, limit: 200}`, `{offset: 201, limit: 200}`, etc.
- You can issue multiple `town_read` calls in the same turn — useful when exploring several files at once after a `town_cp`:
  ```
  town_read  uri: "vfs:///session/files/src/App.tsx"
  town_read  uri: "vfs:///session/files/convex/schema.ts"
  town_read  uri: "vfs:///session/files/convex/auth.ts"  limit: 200
  ```
- When paginating a large file, you can issue all remaining pages together in the same turn once you know `totalLines` from the first read.
- When a tool result is truncated (you see `_truncated: true`): use `town_read uri: "toolresult://[eventId]" offset: 1 limit: 200` to paginate through it — do NOT copy to sandbox.

**3. Search file contents:**

```
town_grep uri: "vfs:///session/files/project" pattern: "myFunction"
```

**4. Write and edit files in VFS:**

```
town_write uri: "vfs:///session/files/project/src/App.tsx" content: "..."
```

For targeted edits (preserving surrounding code):

```
town_write uri: "vfs:///session/files/project/src/App.tsx" old_string: "..." new_string: "..."
```

**5. Create a feature branch:**

```
github_create_branch repo: "owner/repo" branch: "feature/my-feature" from: "main"
```

**6. Commit all changed files atomically (one commit — do NOT use `github_create_or_update_file` in a loop):**

```
github_commit_files
  repo: "owner/repo"
  branch: "feature/my-feature"
  message: "Add authentication flow"
  files: [
    { source_uri: "vfs:///session/files/project/src/Auth.tsx", github_path: "src/Auth.tsx" },
    { source_uri: "vfs:///session/files/project/src/App.tsx",  github_path: "src/App.tsx" }
  ]
```

**7. Open a pull request, then clean up the branch after it merges:**

```
github_create_pull_request repo: "owner/repo" head: "feature/my-feature" base: "main" title: "..." body: "..."
github_delete_branch repo: "owner/repo" branch: "feature/my-feature"
```
