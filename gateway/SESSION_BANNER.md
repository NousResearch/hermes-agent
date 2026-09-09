# Reset banner metadata (draft)

Reset notices retain their model/provider/context block and tip, and add the
serving profile, Hermes package version, profile reasoning default, requested
service tier, delegation defaults and tool approval mode. Runtime YOLO overrides
are shown explicitly. Metadata reads take place inside the serving profile scope;
no new credential resolution, inference, or network probes are introduced.

Delegation inheritance means the main agent at spawn time, not named-profile
workers or task-specific routes. Service tier is a request, not a service guarantee.
Reasoning here is the Hermes configuration level, not provider wire clamping.

This draft is incomplete: it does not display the new session ID, does not resolve
reasoning against channel/fallback model routes, and needs reset-handler lifecycle
coverage (existing and absent sessions, cleared overrides), approval-policy tests,
and identifier usability tests before upstream publication.
