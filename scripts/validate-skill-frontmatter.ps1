[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$SkillsRoot,

    [string[]]$SkillName
)

$ErrorActionPreference = 'Stop'

function Test-SkillFile {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Path,

        [Parameter(Mandatory=$true)]
        [string]$ExpectedName
    )

    if(-not (Test-Path -LiteralPath $Path -PathType Leaf)){
        return [PSCustomObject]@{
            Path = $Path
            Skill = $ExpectedName
            Valid = $false
            Reason = 'SKILL.md not found'
        }
    }

    $bytes = [IO.File]::ReadAllBytes($Path)
    $text = [Text.Encoding]::UTF8.GetString($bytes)
    $lines = $text -split '\r?\n'

    if($text.Contains('\n')){
        return [PSCustomObject]@{
            Path = $Path
            Skill = $ExpectedName
            Valid = $false
            Reason = 'literal backslash-n sequence found'
        }
    }

    $starts = $lines.Count -gt 0 -and $lines[0] -ceq '---'
    $closingIndex = -1

    for($i = 1; $i -lt $lines.Count; $i++){
        if($lines[$i] -ceq '---'){
            $closingIndex = $i
            break
        }
    }

    $nameLine = $lines |
        Where-Object { $_ -match '^name:\s*\S+' } |
        Select-Object -First 1

    $descriptionLine = $lines |
        Where-Object { $_ -match '^description:\s*\S+' } |
        Select-Object -First 1

    $declaredName = ''

    if($nameLine){
        $declaredName = ($nameLine -replace '^name:\s*','').Trim()
    }

    $bodyPresent = $false

    if($closingIndex -ge 0 -and $closingIndex + 1 -lt $lines.Count){
        $body = ($lines[($closingIndex + 1)..($lines.Count - 1)] -join "`n").Trim()
        $bodyPresent = $body.Length -gt 0
    }

    [PSCustomObject]@{
        Path = $Path
        Skill = $ExpectedName
        Valid = (
            $starts -and
            $closingIndex -gt 0 -and
            $null -ne $nameLine -and
            $null -ne $descriptionLine -and
            $declaredName -ceq $ExpectedName -and
            $bodyPresent
        )
        StartsWithDelimiter = $starts
        FrontmatterClosed = ($closingIndex -gt 0)
        NameMatches = ($declaredName -ceq $ExpectedName)
        DescriptionPresent = ($null -ne $descriptionLine)
        BodyPresent = $bodyPresent
        LiteralBackslashN = $text.Contains('\n')
        SHA256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    }
}

if(-not $SkillName -or $SkillName.Count -eq 0){
    $files = Get-ChildItem -LiteralPath $SkillsRoot -Recurse -Filter 'SKILL.md' -File
} else {
    $files = foreach($name in $SkillName){
        $path = Join-Path (Join-Path $SkillsRoot $name) 'SKILL.md'

        if(-not (Test-Path -LiteralPath $path -PathType Leaf)){
            throw "Skill non trovata: $name"
        }

        Get-Item -LiteralPath $path
    }
}

$results = foreach($file in $files){
    $name = Split-Path (Split-Path $file.FullName -Parent) -Leaf
    Test-SkillFile -Path $file.FullName -ExpectedName $name
}

$results | Format-Table -AutoSize

if($results.Valid -contains $false){
    exit 1
}
