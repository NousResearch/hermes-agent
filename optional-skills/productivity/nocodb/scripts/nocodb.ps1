###############################################################################
# nocodb.ps1 - NocoDB v3 CLI (PowerShell)
#
# Vendored verbatim from https://github.com/nocodb/agent-skills (MIT).
# Fixes belong upstream first; re-vendor rather than patching in place.
###############################################################################
$ErrorActionPreference = "Stop"

$NC_URL = if ($env:NOCODB_URL) { $env:NOCODB_URL } else { "https://app.nocodb.com" }
$NC_TOKEN = $env:NOCODB_TOKEN
$NC_VERBOSE = $env:NOCODB_VERBOSE -eq "1"

if (-not $NC_TOKEN) { Write-Error "NOCODB_TOKEN required"; exit 1 }

function _v($msg) { if ($NC_VERBOSE) { Write-Host "→ $msg" -ForegroundColor DarkGray } }

###############################################################################
# HTTP helpers
###############################################################################
$headers = @{ "xc-token" = $NC_TOKEN; "Content-Type" = "application/json" }

function _get($path) {
    Invoke-RestMethod -Uri "$NC_URL/api/v3/$path" -Headers $headers -Method Get
}
function _post($path, $body) {
    Invoke-RestMethod -Uri "$NC_URL/api/v3/$path" -Headers $headers -Method Post -Body $body
}
function _patch($path, $body) {
    Invoke-RestMethod -Uri "$NC_URL/api/v3/$path" -Headers $headers -Method Patch -Body $body
}
function _put($path, $body) {
    Invoke-RestMethod -Uri "$NC_URL/api/v3/$path" -Headers $headers -Method Put -Body $body
}
function _delete($path, $body) {
    $params = @{ Uri = "$NC_URL/api/v3/$path"; Headers = $headers; Method = "Delete" }
    if ($body) { $params.Body = $body }
    Invoke-RestMethod @params
}
function _upload($path, $filePath) {
    $form = @{ file = Get-Item $filePath }
    Invoke-RestMethod -Uri "$NC_URL/api/v3/$path" -Headers @{ "xc-token" = $NC_TOKEN } -Method Post -Form $form
}

function _enc($s) { [System.Uri]::EscapeDataString($s) }

function _jqf($list, $name) {
    $lower = $name.ToLower()
    $match = $list | Where-Object { $_.title.ToLower() -eq $lower } | Select-Object -First 1
    if ($match) { return $match.id }
    return $null
}

###############################################################################
# Validation helpers
###############################################################################
function _err($msg) { Write-Error "error: $msg"; exit 1 }

function _require($name, $val) {
    if (-not $val) { _err "$name is required" }
}

function _require_json($name, $val) {
    if (-not $val) { _err "$name is required" }
    try { $null = $val | ConvertFrom-Json } catch { _err "$name must be valid JSON" }
}

function _require_json_obj($name, $val) {
    if (-not $val) { _err "$name is required" }
    try { $obj = $val | ConvertFrom-Json; if ($obj -is [array]) { _err "$name must be a JSON object" } } catch { _err "$name must be valid JSON" }
}

function _require_json_arr($name, $val) {
    if (-not $val) { _err "$name is required" }
    try { $obj = $val | ConvertFrom-Json; if ($obj -isnot [array]) { _err "$name must be a JSON array" } } catch { _err "$name must be valid JSON" }
}

function _require_int($name, $val) {
    if (-not $val) { return }
    if ($val -notmatch '^\d+$') { _err "$name must be a positive integer" }
}

function _require_nonempty($name, $val) {
    if (-not $val) { _err "$name is required and cannot be empty" }
}

function _require_file($path) {
    if (-not $path) { _err "file path is required" }
    if (-not (Test-Path $path)) { _err "file not found: $path" }
}

function _validate_where($val) {
    if (-not $val) { return }
    if ($val -notmatch '^\(.*\)$' -and $val -notmatch '^~not\(') {
        Write-Warning "where clause should be in format (field,op,value)"
        Write-Warning "  Examples: (name,eq,John) (status,in,active,pending) (age,gte,18)"
        Write-Warning "  Combine:  (field1,eq,x)~and(field2,eq,y)  or  (f1,eq,x)~or(f2,eq,y)"
    }
    if ($val -match '\)and\(' -or $val -match '\)or\(') {
        _err "use ~and and ~or (with tilde), not plain 'and'/'or'. Example: (a,eq,1)~and(b,eq,2)"
    }
}

function _validate_workspace_json($json, $op) {
    _require_json_obj "json" $json
    if ($op -eq "create") {
        $obj = $json | ConvertFrom-Json
        if (-not $obj.title) { _err "json must contain 'title' field" }
    }
}

function _validate_base_json($json, $op) {
    _require_json_obj "json" $json
    if ($op -eq "create") {
        $obj = $json | ConvertFrom-Json
        if (-not $obj.title) { _err "json must contain 'title' field" }
    }
}

function _validate_table_json($json, $op) {
    _require_json_obj "json" $json
    if ($op -eq "create") {
        $obj = $json | ConvertFrom-Json
        if (-not $obj.title) { _err "json must contain 'title' field" }
    }
}

function _validate_field_json($json, $op) {
    _require_json_obj "json" $json
    if ($op -eq "create") {
        $obj = $json | ConvertFrom-Json
        if (-not $obj.title) { _err "json must contain 'title' field" }
        if (-not $obj.type) { _err "json must contain 'type' field" }
    }
}

function _validate_view_json($json, $op) {
    _require_json_obj "json" $json
    if ($op -eq "create") {
        $obj = $json | ConvertFrom-Json
        if (-not $obj.title) { _err "json must contain 'title' field" }
        if (-not $obj.type) { _err "json must contain 'type' field (grid, gallery, form, kanban, calendar)" }
    }
}

function _validate_filter_json($json) {
    _require_json_obj "json" $json
    $obj = $json | ConvertFrom-Json
    if (-not $obj.group_operator) {
        if (-not $obj.field_id) { _err "json must contain 'field_id'" }
        if (-not $obj.operator) { _err "json must contain 'operator' (eq|neq|gt|lt|gte|lte|like|nlike|is|isnot|empty|notempty|null|notnull)" }
    }
}

function _validate_sort_json($json) {
    _require_json_obj "json" $json
    $obj = $json | ConvertFrom-Json
    if (-not $obj.field_id) { _err "json must contain 'field_id'" }
    if ($obj.direction -and $obj.direction -ne "asc" -and $obj.direction -ne "desc") {
        _err "direction must be 'asc' or 'desc'"
    }
}

function _validate_record_json($json) {
    _require_json "json" $json
    $obj = $json | ConvertFrom-Json
    if ($obj -is [array]) {
        if ($obj.Count -eq 0) { _err "json array must not be empty" }
    }
}

function _validate_link_json($json) {
    _require_json "json" $json
    $obj = $json | ConvertFrom-Json
    if ($obj -isnot [array] -or -not $obj[0].id) {
        _err 'json must be array of objects with ''id'' field, e.g. [{"id":42}]'
    }
}

function _validate_members_json($json) {
    _require_json "json" $json
    $obj = $json | ConvertFrom-Json
    if (-not ($obj.email -or $obj.emails -or ($obj -is [array] -and $obj[0].email))) {
        _err "json must contain email(s)"
    }
}

###############################################################################
# ID resolvers
###############################################################################
function _is_ws_id($id) { $id -match '^w[a-z0-9]+$' }
function _is_base_id($id) { $id -match '^p[a-z0-9]+$' }
function _is_tbl_id($id) { $id -match '^m[a-z0-9]+$' }
function _is_view_id($id) { $id -match '^vw[a-z0-9]+$' }
function _is_fld_id($id) { $id -match '^c[a-z0-9]+$' }

function _ws($name) {
    if (_is_ws_id $name) { _v "workspace: $name"; return $name }
    $r = _jqf (_get "meta/workspaces").list $name
    if ($r) { _v "workspace: $name → $r"; return $r }
    _err "workspace not found: $name"
}

function _base($name) {
    if (_is_base_id $name) { _v "base: $name"; return $name }
    $wl = (_get "meta/workspaces").list
    foreach ($w in $wl) {
        $bl = (_get "meta/workspaces/$($w.id)/bases").list
        $r = _jqf $bl $name
        if ($r) { _v "base: $name → $r"; return $r }
    }
    _err "base not found: $name"
}

function _tbl($baseId, $name) {
    if (_is_tbl_id $name) { _v "table: $name"; return $name }
    $r = _jqf (_get "meta/bases/$baseId/tables").list $name
    if ($r) { _v "table: $name → $r"; return $r }
    _err "table not found: $name"
}

function _view($baseId, $tableId, $name) {
    if (_is_view_id $name) { _v "view: $name"; return $name }
    $r = _jqf (_get "meta/bases/$baseId/tables/$tableId/views").list $name
    if ($r) { _v "view: $name → $r"; return $r }
    _err "view not found: $name"
}

function _fld($baseId, $tableId, $name) {
    if (_is_fld_id $name) { _v "field: $name"; return $name }
    $lower = $name.ToLower()
    $tbl = _get "meta/bases/$baseId/tables/$tableId"
    $match = $tbl.fields | Where-Object { $_.title.ToLower() -eq $lower } | Select-Object -First 1
    if ($match) { _v "field: $name → $($match.id)"; return $match.id }
    _err "field not found: $name"
}

###############################################################################
# Output helper
###############################################################################
function _json($obj) { $obj | ConvertTo-Json -Depth 20 }
function _tsv($list, $fields) {
    foreach ($item in $list) {
        ($fields | ForEach-Object { $item.$_ }) -join "`t"
    }
}

###############################################################################
# Commands
###############################################################################
$cmd = $args[0]
$a = $args[1..($args.Length)]

switch ($cmd) {

    #=========================================================================
    # WORKSPACES
    #=========================================================================
    "workspace:list" {
        _tsv (_get "meta/workspaces").list @("title","id")
    }
    "workspace:get" {
        _require "workspace" $a[0]
        _json (_get "meta/workspaces/$(_ws $a[0])")
    }
    "workspace:create" {
        _require "json" $a[0]
        _validate_workspace_json $a[0] "create"
        _json (_post "meta/workspaces" $a[0])
    }
    "workspace:update" {
        _require "workspace" $a[0]; _require "json" $a[1]
        _validate_workspace_json $a[1] "update"
        _json (_patch "meta/workspaces/$(_ws $a[0])" $a[1])
    }
    "workspace:delete" {
        _require "workspace" $a[0]
        _json (_delete "meta/workspaces/$(_ws $a[0])")
    }
    "workspace:members" {
        _require "workspace" $a[0]
        _json (_get "meta/workspaces/$(_ws $a[0])?include[]=members").members
    }
    "workspace:members:add" {
        _require "workspace" $a[0]; _require "json" $a[1]
        _validate_members_json $a[1]
        _json (_post "meta/workspaces/$(_ws $a[0])/members" $a[1])
    }
    "workspace:members:update" {
        _require "workspace" $a[0]; _require "json" $a[1]
        _validate_members_json $a[1]
        _json (_patch "meta/workspaces/$(_ws $a[0])/members" $a[1])
    }
    "workspace:members:remove" {
        _require "workspace" $a[0]; _require "json" $a[1]
        _validate_members_json $a[1]
        _json (_delete "meta/workspaces/$(_ws $a[0])/members" $a[1])
    }

    #=========================================================================
    # BASES
    #=========================================================================
    "base:list" {
        _require "workspace" $a[0]
        _tsv (_get "meta/workspaces/$(_ws $a[0])/bases").list @("title","id")
    }
    "base:get" {
        _require "base" $a[0]
        _json (_get "meta/bases/$(_base $a[0])")
    }
    "base:create" {
        _require "workspace" $a[0]; _require "json" $a[1]
        _validate_base_json $a[1] "create"
        _json (_post "meta/workspaces/$(_ws $a[0])/bases" $a[1])
    }
    "base:update" {
        _require "base" $a[0]; _require "json" $a[1]
        _validate_base_json $a[1] "update"
        _json (_patch "meta/bases/$(_base $a[0])" $a[1])
    }
    "base:delete" {
        _require "base" $a[0]
        _json (_delete "meta/bases/$(_base $a[0])")
    }
    "base:members" {
        _require "base" $a[0]
        _json (_get "meta/bases/$(_base $a[0])?include[]=members").members
    }
    "base:members:add" {
        _require "base" $a[0]; _require "json" $a[1]
        _validate_members_json $a[1]
        _json (_post "meta/bases/$(_base $a[0])/members" $a[1])
    }
    "base:members:update" {
        _require "base" $a[0]; _require "json" $a[1]
        _validate_members_json $a[1]
        _json (_patch "meta/bases/$(_base $a[0])/members" $a[1])
    }
    "base:members:remove" {
        _require "base" $a[0]; _require "json" $a[1]
        _validate_members_json $a[1]
        _json (_delete "meta/bases/$(_base $a[0])/members" $a[1])
    }

    #=========================================================================
    # TABLES
    #=========================================================================
    "table:list" {
        _require "base" $a[0]
        $b = _base $a[0]
        _tsv (_get "meta/bases/$b/tables").list @("title","id")
    }
    "table:get" {
        _require "base" $a[0]; _require "table" $a[1]
        $b = _base $a[0]; $t = _tbl $b $a[1]
        _json (_get "meta/bases/$b/tables/$t")
    }
    "table:create" {
        _require "base" $a[0]; _require "json" $a[1]
        _validate_table_json $a[1] "create"
        _json (_post "meta/bases/$(_base $a[0])/tables" $a[1])
    }
    "table:update" {
        _require "base" $a[0]; _require "table" $a[1]; _require "json" $a[2]
        _validate_table_json $a[2] "update"
        $b = _base $a[0]; $t = _tbl $b $a[1]
        _json (_patch "meta/bases/$b/tables/$t" $a[2])
    }
    "table:delete" {
        _require "base" $a[0]; _require "table" $a[1]
        $b = _base $a[0]; $t = _tbl $b $a[1]
        _json (_delete "meta/bases/$b/tables/$t")
    }

    #=========================================================================
    # FIELDS
    #=========================================================================
    "field:list" {
        _require "base" $a[0]; _require "table" $a[1]
        $b = _base $a[0]; $t = _tbl $b $a[1]
        _tsv (_get "meta/bases/$b/tables/$t").fields @("title","type","id")
    }
    "field:get" {
        _require "base" $a[0]; _require "table" $a[1]; _require "field" $a[2]
        $b = _base $a[0]; $t = _tbl $b $a[1]; $f = _fld $b $t $a[2]
        _json (_get "meta/bases/$b/fields/$f")
    }
    "field:create" {
        _require "base" $a[0]; _require "table" $a[1]; _require "json" $a[2]
        _validate_field_json $a[2] "create"
        $b = _base $a[0]; $t = _tbl $b $a[1]
        _json (_post "meta/bases/$b/tables/$t/fields" $a[2])
    }
    "field:update" {
        _require "base" $a[0]; _require "table" $a[1]; _require "field" $a[2]; _require "json" $a[3]
        _validate_field_json $a[3] "update"
        $b = _base $a[0]; $t = _tbl $b $a[1]; $f = _fld $b $t $a[2]
        _json (_patch "meta/bases/$b/fields/$f" $a[3])
    }
    "field:delete" {
        _require "base" $a[0]; _require "table" $a[1]; _require "field" $a[2]
        $b = _base $a[0]; $t = _tbl $b $a[1]; $f = _fld $b $t $a[2]
        _json (_delete "meta/bases/$b/fields/$f")
    }

    #=========================================================================
    # VIEWS
    #=========================================================================
    "view:list" {
        _require "base" $a[0]; _require "table" $a[1]
        $b = _base $a[0]; $t = _tbl $b $a[1]
        _tsv (_get "meta/bases/$b/tables/$t/views").list @("title","type","id")
    }
    "view:get" {
        _require "base" $a[0]; _require "table" $a[1]; _require "view" $a[2]
        $b = _base $a[0]; $t = _tbl $b $a[1]; $v = _view $b $t $a[2]
        _json (_get "meta/bases/$b/views/$v")
    }
    "view:create" {
        _require "base" $a[0]; _require "table" $a[1]; _require "json" $a[2]
        _validate_view_json $a[2] "create"
        $b = _base $a[0]; $t = _tbl $b $a[1]
        _json (_post "meta/bases/$b/tables/$t/views" $a[2])
    }
    "view:update" {
        _require "base" $a[0]; _require "table" $a[1]; _require "view" $a[2]; _require "json" $a[3]
        _validate_view_json $a[3] "update"
        $b = _base $a[0]; $t = _tbl $b $a[1]; $v = _view $b $t $a[2]
        _json (_patch "meta/bases/$b/views/$v" $a[3])
    }
    "view:delete" {
        _require "base" $a[0]; _require "table" $a[1]; _require "view" $a[2]
        $b = _base $a[0]; $t = _tbl $b $a[1]; $v = _view $b $t $a[2]
        _json (_delete "meta/bases/$b/views/$v")
    }

    #=========================================================================
    # VIEW FILTERS
    #=========================================================================
    "filter:list" {
        _require "base" $a[0]; _require "table" $a[1]; _require "view" $a[2]
        $b = _base $a[0]; $t = _tbl $b $a[1]; $v = _view $b $t $a[2]
        _json (_get "meta/bases/$b/views/$v/filters")
    }
    "filter:create" {
        _require "base" $a[0]; _require "table" $a[1]; _require "view" $a[2]; _require "json" $a[3]
        _validate_filter_json $a[3]
        $b = _base $a[0]; $t = _tbl $b $a[1]; $v = _view $b $t $a[2]
        _json (_post "meta/bases/$b/views/$v/filters" $a[3])
    }
    "filter:replace" {
        _require "base" $a[0]; _require "table" $a[1]; _require "view" $a[2]; _require "json" $a[3]
        _require_json "json" $a[3]
        $b = _base $a[0]; $t = _tbl $b $a[1]; $v = _view $b $t $a[2]
        _json (_put "meta/bases/$b/views/$v/filters" $a[3])
    }
    "filter:update" {
        _require "base" $a[0]; _require "filterId" $a[1]; _require "json" $a[2]
        _validate_filter_json $a[2]
        _json (_patch "meta/bases/$(_base $a[0])/filters/$($a[1])" $a[2])
    }
    "filter:delete" {
        _require "base" $a[0]; _require "filterId" $a[1]
        _json (_delete "meta/bases/$(_base $a[0])/filters/$($a[1])")
    }

    #=========================================================================
    # VIEW SORTS
    #=========================================================================
    "sort:list" {
        _require "base" $a[0]; _require "table" $a[1]; _require "view" $a[2]
        $b = _base $a[0]; $t = _tbl $b $a[1]; $v = _view $b $t $a[2]
        _json (_get "meta/bases/$b/views/$v/sorts")
    }
    "sort:create" {
        _require "base" $a[0]; _require "table" $a[1]; _require "view" $a[2]; _require "json" $a[3]
        _validate_sort_json $a[3]
        $b = _base $a[0]; $t = _tbl $b $a[1]; $v = _view $b $t $a[2]
        _json (_post "meta/bases/$b/views/$v/sorts" $a[3])
    }
    "sort:update" {
        _require "base" $a[0]; _require "sortId" $a[1]; _require "json" $a[2]
        _validate_sort_json $a[2]
        _json (_patch "meta/bases/$(_base $a[0])/sorts/$($a[1])" $a[2])
    }
    "sort:delete" {
        _require "base" $a[0]; _require "sortId" $a[1]
        _json (_delete "meta/bases/$(_base $a[0])/sorts/$($a[1])")
    }

    #=========================================================================
    # RECORDS
    #=========================================================================
    "record:list" {
        _require "base" $a[0]; _require "table" $a[1]
        _require_int "page" $a[2]; _require_int "size" $a[3]
        _validate_where $a[4]; _require_int "nestedPage" $a[8]
        $b = _base $a[0]; $t = _tbl $b $a[1]
        $pg = if ($a[2]) { $a[2] } else { "1" }
        $sz = if ($a[3]) { $a[3] } else { "25" }
        $wh = $a[4]; $so = $a[5]; $fl = $a[6]; $vi = $a[7]; $np = $a[8]
        $q = "page=$pg&pageSize=$sz"
        if ($wh) { $q += "&where=$(_enc $wh)" }
        if ($so) { $q += "&sort=$(_enc $so)" }
        if ($fl) { $q += "&fields=$(_enc $fl)" }
        if ($vi) { $q += "&viewId=$vi" }
        if ($np) { $q += "&nestedPage=$np" }
        _json (_get "data/$b/$t/records?$q").records
    }
    "record:get" {
        _require "base" $a[0]; _require "table" $a[1]; _require_nonempty "id" $a[2]
        $b = _base $a[0]; $t = _tbl $b $a[1]
        $q = ""; if ($a[3]) { $q = "?fields=$(_enc $a[3])" }
        _json (_get "data/$b/$t/records/$($a[2])$q")
    }
    "record:create" {
        _require "base" $a[0]; _require "table" $a[1]; _require "json" $a[2]
        _validate_record_json $a[2]
        $b = _base $a[0]; $t = _tbl $b $a[1]
        _json (_post "data/$b/$t/records" $a[2])
    }
    "record:update" {
        _require "base" $a[0]; _require "table" $a[1]; _require_nonempty "id" $a[2]; _require "json" $a[3]
        _require_json_obj "json" $a[3]
        $b = _base $a[0]; $t = _tbl $b $a[1]
        $body = "[{`"id`":$($a[2]),`"fields`":$($a[3])}]"
        $result = _patch "data/$b/$t/records" $body
        _json $result.records[0]
    }
    "record:update-many" {
        _require "base" $a[0]; _require "table" $a[1]; _require "json" $a[2]
        _require_json_arr "json" $a[2]
        $b = _base $a[0]; $t = _tbl $b $a[1]
        _json (_patch "data/$b/$t/records" $a[2])
    }
    "record:delete" {
        _require "base" $a[0]; _require "table" $a[1]; _require "id" $a[2]
        $b = _base $a[0]; $t = _tbl $b $a[1]
        if ($a[2] -match '^\[') {
            try {
                $arr = $a[2] | ConvertFrom-Json
                if ($arr[0].id) { $ids = $a[2] }
                else { $ids = ($arr | ForEach-Object { "{`"id`":`"$_`"}" }) -join ","; $ids = "[$ids]" }
            } catch { _err "array must contain strings or objects with 'id' field" }
        } else {
            _require_nonempty "id" $a[2]
            $ids = "[{`"id`":`"$($a[2])`"}]"
        }
        _json (_delete "data/$b/$t/records" $ids)
    }
    "record:count" {
        _require "base" $a[0]; _require "table" $a[1]
        _validate_where $a[2]
        $b = _base $a[0]; $t = _tbl $b $a[1]
        $q = @()
        if ($a[2]) { $q += "where=$(_enc $a[2])" }
        if ($a[3]) { $q += "viewId=$($a[3])" }
        $qs = if ($q.Count -gt 0) { "?" + ($q -join "&") } else { "" }
        (_get "data/$b/$t/count$qs").count
    }

    #=========================================================================
    # LINKED RECORDS
    #=========================================================================
    "link:list" {
        _require "base" $a[0]; _require "table" $a[1]; _require "linkField" $a[2]; _require_nonempty "recordId" $a[3]
        _require_int "page" $a[4]; _require_int "size" $a[5]; _validate_where $a[6]
        $b = _base $a[0]; $t = _tbl $b $a[1]; $f = _fld $b $t $a[2]
        $pg = if ($a[4]) { $a[4] } else { "1" }
        $sz = if ($a[5]) { $a[5] } else { "25" }
        $q = "page=$pg&pageSize=$sz"
        if ($a[6]) { $q += "&where=$(_enc $a[6])" }
        if ($a[7]) { $q += "&sort=$(_enc $a[7])" }
        if ($a[8]) { $q += "&fields=$(_enc $a[8])" }
        _json (_get "data/$b/$t/links/$f/$($a[3])?$q")
    }
    "link:add" {
        _require "base" $a[0]; _require "table" $a[1]; _require "linkField" $a[2]; _require_nonempty "recordId" $a[3]; _require "json" $a[4]
        _validate_link_json $a[4]
        $b = _base $a[0]; $t = _tbl $b $a[1]; $f = _fld $b $t $a[2]
        _json (_post "data/$b/$t/links/$f/$($a[3])" $a[4])
    }
    "link:remove" {
        _require "base" $a[0]; _require "table" $a[1]; _require "linkField" $a[2]; _require_nonempty "recordId" $a[3]; _require "json" $a[4]
        _validate_link_json $a[4]
        $b = _base $a[0]; $t = _tbl $b $a[1]; $f = _fld $b $t $a[2]
        _json (_delete "data/$b/$t/links/$f/$($a[3])" $a[4])
    }

    #=========================================================================
    # ATTACHMENTS
    #=========================================================================
    "attachment:upload" {
        _require "base" $a[0]; _require "table" $a[1]; _require_nonempty "recordId" $a[2]; _require "field" $a[3]; _require_file $a[4]
        $b = _base $a[0]; $t = _tbl $b $a[1]; $f = _fld $b $t $a[3]
        _json (_upload "data/$b/$t/records/$($a[2])/fields/$f/upload" $a[4])
    }

    #=========================================================================
    # BUTTON ACTIONS
    #=========================================================================
    "action:trigger" {
        _require "base" $a[0]; _require "table" $a[1]; _require "buttonField" $a[2]; _require_nonempty "recordId" $a[3]
        $b = _base $a[0]; $t = _tbl $b $a[1]; $f = _fld $b $t $a[2]
        _json (_post "data/$b/$t/actions/$f" "{`"recordId`":`"$($a[3])`"}")
    }

    #=========================================================================
    # SCRIPTS
    #=========================================================================
    "script:list" {
        _require "base" $a[0]
        _tsv (_get "meta/bases/$(_base $a[0])/scripts").list @("title","id")
    }
    "script:get" {
        _require "base" $a[0]; _require "scriptId" $a[1]
        _json (_get "meta/bases/$(_base $a[0])/scripts/$($a[1])")
    }
    "script:create" {
        _require "base" $a[0]; _require "json" $a[1]
        _require_json_obj "json" $a[1]
        $obj = $a[1] | ConvertFrom-Json
        if (-not $obj.title) { _err "json must contain 'title' field" }
        _json (_post "meta/bases/$(_base $a[0])/scripts" $a[1])
    }
    "script:update" {
        _require "base" $a[0]; _require "scriptId" $a[1]; _require "json" $a[2]
        _require_json_obj "json" $a[2]
        _json (_patch "meta/bases/$(_base $a[0])/scripts/$($a[1])" $a[2])
    }
    "script:delete" {
        _require "base" $a[0]; _require "scriptId" $a[1]
        _json (_delete "meta/bases/$(_base $a[0])/scripts/$($a[1])")
    }

    #=========================================================================
    # TEAMS
    #=========================================================================
    "team:list" {
        _require "workspace" $a[0]
        _tsv (_get "meta/workspaces/$(_ws $a[0])/teams").list @("title","id")
    }
    "team:get" {
        _require "workspace" $a[0]; _require "teamId" $a[1]
        _json (_get "meta/workspaces/$(_ws $a[0])/teams/$($a[1])")
    }
    "team:create" {
        _require "workspace" $a[0]; _require "json" $a[1]
        _require_json_obj "json" $a[1]
        $obj = $a[1] | ConvertFrom-Json
        if (-not $obj.title) { _err "json must contain 'title' field" }
        _json (_post "meta/workspaces/$(_ws $a[0])/teams" $a[1])
    }
    "team:update" {
        _require "workspace" $a[0]; _require "teamId" $a[1]; _require "json" $a[2]
        _require_json_obj "json" $a[2]
        _json (_patch "meta/workspaces/$(_ws $a[0])/teams/$($a[1])" $a[2])
    }
    "team:delete" {
        _require "workspace" $a[0]; _require "teamId" $a[1]
        _json (_delete "meta/workspaces/$(_ws $a[0])/teams/$($a[1])")
    }
    "team:members:add" {
        _require "workspace" $a[0]; _require "teamId" $a[1]; _require "json" $a[2]
        _validate_members_json $a[2]
        _json (_post "meta/workspaces/$(_ws $a[0])/teams/$($a[1])/members" $a[2])
    }
    "team:members:update" {
        _require "workspace" $a[0]; _require "teamId" $a[1]; _require "json" $a[2]
        _require_json "json" $a[2]
        _json (_patch "meta/workspaces/$(_ws $a[0])/teams/$($a[1])/members" $a[2])
    }
    "team:members:remove" {
        _require "workspace" $a[0]; _require "teamId" $a[1]; _require "json" $a[2]
        _require_json "json" $a[2]
        _json (_delete "meta/workspaces/$(_ws $a[0])/teams/$($a[1])/members" $a[2])
    }

    #=========================================================================
    # API TOKENS
    #=========================================================================
    "token:list" {
        _tsv (_get "meta/tokens").list @("title","id")
    }
    "token:create" {
        _require "json" $a[0]
        _require_json_obj "json" $a[0]
        $obj = $a[0] | ConvertFrom-Json
        if (-not $obj.title) { _err "json must contain 'title' field" }
        _json (_post "meta/tokens" $a[0])
    }
    "token:delete" {
        _require "tokenId" $a[0]
        _json (_delete "meta/tokens/$($a[0])")
    }

    #=========================================================================
    # WHERE FILTER HELP
    #=========================================================================
    { $_ -eq "where:help" -or $_ -eq "filter:help" } {
        @"
WHERE FILTER SYNTAX
===================

BASIC SYNTAX:
  (field,operator,value)           Basic filter
  (field,operator)                 No-value operators (blank, null, checked, etc.)
  (field,op,sub_op)                Date with sub-operator
  (field,op,sub_op,value)          Date with sub-operator and value
  (field,operator,val1,val2,val3)  Multiple values

OPERATORS:

  Text/General:
    eq        Equal to                    (name,eq,John)
    neq       Not equal to                (status,neq,archived)
    like      Contains (use % wildcard)   (name,like,%john%)
    nlike     Does not contain            (name,nlike,%test%)
    in        In list of values           (status,in,active,pending,review)

  Numeric:
    gt        Greater than (>)            (price,gt,100)
    lt        Less than (<)               (stock,lt,10)
    gte       Greater than or equal (>=)  (rating,gte,4)
    lte       Less than or equal (<=)     (age,lte,65)

  Range:
    btw       Between (inclusive)         (price,btw,10,100)
    nbtw      Not between                 (score,nbtw,0,50)

  Null/Empty (no value needed):
    blank     Is blank (null or empty)    (notes,blank)
    notblank  Is not blank                (email,notblank)
    null      Is null                     (deleted_at,null)
    notnull   Is not null                 (created_by,notnull)
    empty     Is empty string             (description,empty)
    notempty  Is not empty string         (title,notempty)

  Boolean/Checkbox (no value needed):
    checked      Is checked/true          (is_active,checked)
    notchecked   Is not checked/false     (is_archived,notchecked)

  Multi-Select/Tags:
    allof     Contains all of             (tags,allof,urgent,important)
    anyof     Contains any of             (tags,anyof,bug,feature)
    nallof    Does not contain all of     (tags,nallof,spam,junk)
    nanyof    Does not contain any of     (categories,nanyof,draft,deleted)

DATE/TIME FILTERING:

  isWithin - Date falls within time range:
    Sub-ops (no value): pastWeek, pastMonth, pastYear, nextWeek, nextMonth, nextYear
    Sub-ops (with days): pastNumberOfDays, nextNumberOfDays
    Examples:
      (created_at,isWithin,pastWeek)
      (due_date,isWithin,pastNumberOfDays,14)

  eq, neq, gt, lt, gte, lte - Date comparisons:
    Sub-ops (no value): today, tomorrow, yesterday, oneWeekAgo, oneWeekFromNow
    Sub-ops (with days): daysAgo, daysFromNow
    Sub-op (with YYYY-MM-DD): exactDate
    Examples:
      (created_at,eq,today)
      (due_date,lt,today)                      # Overdue
      (event_date,eq,exactDate,2024-06-15)
      (created_at,gte,daysAgo,7)

  btw - Date range:
      (event_date,btw,2024-01-01,2024-12-31)

LOGICAL OPERATIONS:

  IMPORTANT: Use ~and, ~or, ~not (with tilde prefix)

  AND: (filter1)~and(filter2)
  OR:  (filter1)~or(filter2)
  NOT: ~not(filter)

  Examples:
    (name,eq,John)~and(age,gte,18)
    (status,eq,active)~or(status,eq,pending)
    ~not(is_deleted,checked)
    (status,in,active,pending)~and(country,eq,USA)

SPECIAL VALUES:
  NULL value:         (field,eq,null)
  Empty string:       (field,eq,'') or (field,eq,"")
  Value with comma:   (field,eq,"hello, world")
  Value with quotes:  (field,eq,"it's here")

COMPLEX EXAMPLES:
  Active users this month:
    (status,eq,active)~and(created_at,isWithin,pastMonth)

  Overdue high-priority:
    (due_date,lt,today)~and(priority,eq,high)~and(completed,notchecked)

  Orders `$100-`$500 pending:
    (amount,gte,100)~and(amount,lte,500)~and(status,in,pending,processing)

  Updated recently, not archived:
    (updated_at,isWithin,pastNumberOfDays,14)~and~not(is_archived,checked)
"@
    }

    #=========================================================================
    # HELP
    #=========================================================================
    default {
        @"
nocodb.ps1 - NocoDB v3 CLI (PowerShell)

ARGUMENT ORDER: Commands follow a hierarchical pattern:
  workspace -> base -> table -> view/field -> record

  Most commands accept NAMES or IDs. Use IDs directly for faster execution.
  Set `$env:NOCODB_VERBOSE=1 to see resolved IDs.

WORKSPACES  (Enterprise plans only: self-hosted or cloud-hosted)
  workspace:list
  workspace:get        WORKSPACE
  workspace:create     JSON
  workspace:update     WORKSPACE  JSON
  workspace:delete     WORKSPACE

WORKSPACE COLLABORATION  (self-hosted Enterprise only)
  workspace:members    WORKSPACE
  workspace:members:add/update/remove  WORKSPACE  JSON

BASES
  base:list    WORKSPACE
  base:get     BASE
  base:create  WORKSPACE  JSON
  base:update  BASE  JSON
  base:delete  BASE

BASE COLLABORATION  (Enterprise only: self-hosted or cloud-hosted)
  base:members BASE
  base:members:add/update/remove  BASE  JSON

TABLES
  table:list    BASE
  table:get     BASE  TABLE
  table:create  BASE  JSON
  table:update  BASE  TABLE  JSON
  table:delete  BASE  TABLE

FIELDS
  field:list    BASE  TABLE
  field:get     BASE  TABLE  FIELD
  field:create  BASE  TABLE  JSON
  field:update  BASE  TABLE  FIELD  JSON
  field:delete  BASE  TABLE  FIELD

VIEWS  (Enterprise only: self-hosted or cloud-hosted)
  view:list    BASE  TABLE
  view:get     BASE  TABLE  VIEW
  view:create  BASE  TABLE  JSON
  view:update  BASE  TABLE  VIEW  JSON
  view:delete  BASE  TABLE  VIEW

FILTERS
  filter:list     BASE  TABLE  VIEW
  filter:create   BASE  TABLE  VIEW  JSON
  filter:replace  BASE  TABLE  VIEW  JSON
  filter:update   BASE  FILTER_ID  JSON
  filter:delete   BASE  FILTER_ID

SORTS
  sort:list    BASE  TABLE  VIEW
  sort:create  BASE  TABLE  VIEW  JSON
  sort:update  BASE  SORT_ID  JSON
  sort:delete  BASE  SORT_ID

RECORDS
  record:list        BASE  TABLE  [PAGE] [SIZE] [WHERE] [SORT] [FIELDS] [VIEW_ID] [NESTED_PAGE]
  record:get         BASE  TABLE  RECORD_ID  [FIELDS]
  record:create      BASE  TABLE  JSON
  record:update      BASE  TABLE  RECORD_ID  JSON
  record:update-many BASE  TABLE  JSON_ARRAY
  record:delete      BASE  TABLE  RECORD_ID_OR_ARRAY
  record:count       BASE  TABLE  [WHERE] [VIEW_ID]

LINKED RECORDS
  link:list    BASE  TABLE  LINK_FIELD  RECORD_ID  [PAGE] [SIZE] [WHERE] [SORT] [FIELDS]
  link:add     BASE  TABLE  LINK_FIELD  RECORD_ID  JSON_ARRAY
  link:remove  BASE  TABLE  LINK_FIELD  RECORD_ID  JSON_ARRAY

ATTACHMENTS
  attachment:upload  BASE  TABLE  RECORD_ID  FIELD  FILEPATH

BUTTON ACTIONS
  action:trigger  BASE  TABLE  BUTTON_FIELD  RECORD_ID

SCRIPTS  (Enterprise only: self-hosted or cloud-hosted)
  script:list    BASE
  script:get     BASE  SCRIPT_ID
  script:create  BASE  JSON
  script:update  BASE  SCRIPT_ID  JSON
  script:delete  BASE  SCRIPT_ID

TEAMS  (Enterprise only: self-hosted or cloud-hosted)
  team:list    WORKSPACE
  team:get     WORKSPACE  TEAM_ID
  team:create  WORKSPACE  JSON
  team:update  WORKSPACE  TEAM_ID  JSON
  team:delete  WORKSPACE  TEAM_ID
  team:members:add/update/remove  WORKSPACE  TEAM_ID  JSON

API TOKENS  (Enterprise only: self-hosted or cloud-hosted)
  token:list
  token:create  JSON
  token:delete  TOKEN_ID

WHERE HELP
  where:help    Show where filter syntax and operators

ENVIRONMENT
  `$env:NOCODB_URL      Base URL (default: https://app.nocodb.com)
  `$env:NOCODB_TOKEN    API token (required)
  `$env:NOCODB_VERBOSE  Set to 1 to show resolved IDs

EXAMPLES
  # IDs: w=workspace, p=base, m=table, c=column, vw=view (all lowercase alphanumeric)

  # List workspaces, bases, tables (use IDs from output)
  .\nocodb.ps1 workspace:list                                   # -> wabc1234xyz
  .\nocodb.ps1 base:list wabc1234xyz                            # -> pdef5678uvw
  .\nocodb.ps1 table:list pdef5678uvw                           # -> mghi9012rst
  .\nocodb.ps1 field:list pdef5678uvw mghi9012rst               # -> cjkl3456opq
  .\nocodb.ps1 view:list pdef5678uvw mghi9012rst                # -> vwmno7890abc

  # Records (BASE_ID TABLE_ID ...)
  .\nocodb.ps1 record:list pdef5678uvw mghi9012rst 1 50 "(Status,eq,active)"
  .\nocodb.ps1 record:get pdef5678uvw mghi9012rst 31
  .\nocodb.ps1 record:create pdef5678uvw mghi9012rst '{"fields":{"Name":"Alice"}}'
  .\nocodb.ps1 record:update pdef5678uvw mghi9012rst 31 '{"Status":"done"}'
  .\nocodb.ps1 record:delete pdef5678uvw mghi9012rst 31

  # Names also work (resolved to IDs automatically)
  .\nocodb.ps1 record:list MyBase Users
  `$env:NOCODB_VERBOSE="1"; .\nocodb.ps1 field:list MyBase Users   # shows resolved IDs
"@
    }
}
