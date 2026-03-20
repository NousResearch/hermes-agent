from __future__ import annotations

PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__PRODUCT_NAME__</title>
<style>__PAGE_STYLE__</style>
</head>
<body>
<main>
<div class="topbar">
<div class="brand"><span class="brand-mark"></span><span id="brandName">__PRODUCT_NAME__</span></div>
<div class="top-controls">
<div class="mini-control session-chip" id="sessionChip" hidden></div>
<button type="button" class="mini-control theme-toggle" id="themeToggle" aria-label="Toggle theme">__DARK_ICON__</button>
<button type="button" class="mini-control logout-button" id="logoutButton" hidden>Log out</button>
</div>
</div>

<div class="stack">
<section class="popup hero-card" id="authCard">
<div class="popup-inner hero-inner">
<p class="eyebrow">__PRODUCT_NAME__</p>
<h1 id="heroTitle">Private local agents for your team.</h1>
<p class="lead" id="heroLead">Sign in to open your personal workspace and chat with your agent.</p>
<div class="pill-list">
<span class="pill">Chat</span>
<span class="pill">Private</span>
<span class="pill">Local</span>
</div>
<div class="actions">
<a class="button" id="loginButton" href="/api/auth/login">Sign in with Pocket ID</a>
</div>
<div id="authMessage" class="message"></div>
</div>
</section>

<section class="popup shell-card" id="chatCard" hidden>
<div class="popup-inner">
<div class="section-head">
<div>
<p class="eyebrow">Chat</p>
<h2>Your Agent</h2>
</div>
<span class="status-pill">Streaming UX kept</span>
</div>
<div id="chatMessage" class="message"></div>
<div class="shell">
<div class="chat-log" id="chatLog">
<div class="empty-state">No messages yet.</div>
</div>
<form class="chat-form" id="chatForm">
<label class="field full">
<span>Message</span>
<textarea id="chatInput" placeholder="Write a message to your agent." required></textarea>
</label>
<div class="actions">
<button class="button" id="chatSubmit" type="submit">Send</button>
</div>
</form>
</div>
</div>
</section>

<section class="popup shell-card" id="workspaceCard" hidden>
<div class="popup-inner">
<p class="eyebrow">Workspace</p>
<h2>Shared Files</h2>
<p class="lead small">This area will expose the user workspace that is live-mounted into the isolated runtime.</p>
</div>
</section>

<section class="popup shell-card" id="adminCard" hidden>
<div class="popup-inner">
<div class="section-head">
<div>
<p class="eyebrow">Admin</p>
<h2>User Management</h2>
</div>
<span class="status-pill">Compact by design</span>
</div>
<p class="lead small">Admins will get a narrow user-management surface here for creating and deleting users without exposing broader setup controls in the browser.</p>
<div class="table-shell">
<table>
<thead>
<tr><th>User</th><th>Role</th><th>Status</th><th>Next</th></tr>
</thead>
<tbody id="adminUsersTable">
<tr><td colspan="4" class="muted-cell">Pocket ID-backed user actions will be wired into this panel next.</td></tr>
</tbody>
</table>
</div>
</div>
</section>
</div>
</main>
<script>__PAGE_SCRIPT__</script>
</body>
</html>"""
