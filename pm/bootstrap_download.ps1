param([string]$RequestFile, [string]$ResultFile)
$ErrorActionPreference = 'Stop'
$outcome = @{ ok = $false; error = ''; headers = @{} }
$client = $null
$response = $null
try {
    Add-Type -AssemblyName System.Net.Http
    $request = Get-Content -LiteralPath $RequestFile -Raw -Encoding UTF8 | ConvertFrom-Json
    $handler = New-Object System.Net.Http.HttpClientHandler
    $handler.AllowAutoRedirect = $false
    if ($request.proxy) {
        $handler.Proxy = New-Object System.Net.WebProxy($request.proxy)
        $proxyUri = [Uri]$request.proxy
        if ($proxyUri.UserInfo) {
            $credentials = $proxyUri.UserInfo -split ':', 2
            $password = if ($credentials.Count -eq 2) { [Uri]::UnescapeDataString($credentials[1]) } else { '' }
            $handler.Proxy.Credentials = New-Object System.Net.NetworkCredential([Uri]::UnescapeDataString($credentials[0]), $password)
        }
    } else {
        $handler.UseProxy = $false
    }
    # HttpClient's unmodified verifier uses Windows' certificate chain policy.
    $client = New-Object System.Net.Http.HttpClient($handler)
    $client.Timeout = [TimeSpan]::FromSeconds(120)
    $client.DefaultRequestHeaders.UserAgent.ParseAdd($request.user_agent)
    $client.DefaultRequestHeaders.AcceptEncoding.ParseAdd('identity')
    $uri = [Uri]$request.url
    $loopback = $uri.Scheme -eq 'http' -and $uri.IsLoopback
    for ($hop = 0; $hop -le 10; $hop++) {
        if ($uri.Scheme -ne 'https' -and -not ($loopback -and $uri.Scheme -eq 'http' -and $uri.IsLoopback)) {
            throw 'Refusing redirect to a non-HTTPS URL'
        }
        $response = $client.GetAsync($uri, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
        $status = [int]$response.StatusCode
        if ($status -in @(301, 302, 303, 307, 308)) {
            if (-not $response.Headers.Location -or $hop -eq 10) { throw 'Invalid or excessive HTTPS redirects' }
            $uri = New-Object Uri($uri, $response.Headers.Location)
            $loopback = $loopback -and $uri.Scheme -eq 'http' -and $uri.IsLoopback
            $response.Dispose()
            $response = $null
            continue
        }
        if ($status -ne 200) {
            $outcome.status = $status
            foreach ($header in $response.Headers) { $outcome.headers[$header.Key] = $header.Value -join ', ' }
            throw "HTTP $status"
        }
        if (@($response.Content.Headers.ContentEncoding | Where-Object { $_ -ne 'identity' }).Count) {
            throw 'Encoded response cannot be used as an identity download'
        }
        $incoming = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
        $output = [IO.File]::Open($request.destination, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::Read)
        try { $incoming.CopyTo($output); $output.Flush($true) }
        finally { $output.Dispose(); $incoming.Dispose() }
        $length = $response.Content.Headers.ContentLength
        if ($null -ne $length -and (Get-Item -LiteralPath $request.destination).Length -ne $length) {
            throw 'Incomplete Windows bootstrap download'
        }
        $outcome.ok = $true
        break
    }
} catch {
    $outcome.error = $_.Exception.ToString()
    $cause = $_.Exception
    while ($cause) {
        if ($cause -is [System.Security.Authentication.AuthenticationException]) { $outcome.tls = $true }
        if ($cause -is [System.Threading.Tasks.TaskCanceledException]) { $outcome.timeout = $true }
        if ($cause -is [System.Net.Sockets.SocketException]) { $outcome.connection = $true }
        $cause = $cause.InnerException
    }
} finally {
    if ($response) { $response.Dispose() }
    if ($client) { $client.Dispose() }
    $outcome | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $ResultFile -Encoding UTF8
}
if (-not $outcome.ok) { exit 1 }
