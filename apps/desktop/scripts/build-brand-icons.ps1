param([string]$MasterPath = (Join-Path $PSScriptRoot '..\assets\icon-master.png'))

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$assetDir = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\assets'))
$sourceImage = [Drawing.Image]::FromFile([IO.Path]::GetFullPath($MasterPath))

function Get-IconPng([int]$Size) {
    $bitmap = [Drawing.Bitmap]::new($Size, $Size, [Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $graphics = [Drawing.Graphics]::FromImage($bitmap)
    $stream = [IO.MemoryStream]::new()
    try {
        $graphics.Clear([Drawing.Color]::Transparent)
        $graphics.InterpolationMode = [Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $graphics.PixelOffsetMode = [Drawing.Drawing2D.PixelOffsetMode]::HighQuality
        $graphics.DrawImage($sourceImage, 0, 0, $Size, $Size)
        $bitmap.Save($stream, [Drawing.Imaging.ImageFormat]::Png)
        return ,$stream.ToArray()
    } finally {
        $stream.Dispose()
        $graphics.Dispose()
        $bitmap.Dispose()
    }
}

function Write-BigEndian([IO.BinaryWriter]$Writer, [int]$Value) {
    $bytes = [BitConverter]::GetBytes($Value)
    [Array]::Reverse($bytes)
    $Writer.Write($bytes)
}

try {
    [IO.File]::WriteAllBytes((Join-Path $assetDir 'icon.png'), (Get-IconPng 1024))
    [IO.File]::WriteAllBytes((Join-Path $assetDir '..\public\apple-touch-icon.png'), (Get-IconPng 180))

    $sizes = @(16, 24, 32, 48, 64, 128, 256)
    $payloads = @($sizes | ForEach-Object { ,(Get-IconPng $_) })
    $writer = [IO.BinaryWriter]::new([IO.File]::Create((Join-Path $assetDir 'icon.ico')))
    try {
        $writer.Write([uint16]0)
        $writer.Write([uint16]1)
        $writer.Write([uint16]$sizes.Count)
        $offset = 6 + 16 * $sizes.Count
        for ($i = 0; $i -lt $sizes.Count; $i++) {
            $edge = if ($sizes[$i] -eq 256) { 0 } else { $sizes[$i] }
            $writer.Write([byte]$edge); $writer.Write([byte]$edge)
            $writer.Write([byte]0); $writer.Write([byte]0)
            $writer.Write([uint16]1); $writer.Write([uint16]32)
            $writer.Write([uint32]$payloads[$i].Length)
            $writer.Write([uint32]$offset)
            $offset += $payloads[$i].Length
        }
        foreach ($png in $payloads) { $writer.Write([byte[]]$png) }
    } finally { $writer.Dispose() }

    $chunks = @(@('icp4',16), @('icp5',32), @('icp6',64), @('ic07',128), @('ic08',256), @('ic09',512), @('ic10',1024))
    $buffer = [IO.MemoryStream]::new()
    $writer = [IO.BinaryWriter]::new($buffer)
    try {
        foreach ($chunk in $chunks) {
            $png = Get-IconPng $chunk[1]
            $writer.Write([Text.Encoding]::ASCII.GetBytes($chunk[0]))
            Write-BigEndian $writer ($png.Length + 8)
            $writer.Write([byte[]]$png)
        }
        $body = $buffer.ToArray()
        $output = [IO.BinaryWriter]::new([IO.File]::Create((Join-Path $assetDir 'icon.icns')))
        try {
            $output.Write([Text.Encoding]::ASCII.GetBytes('icns'))
            Write-BigEndian $output ($body.Length + 8)
            $output.Write($body)
        } finally { $output.Dispose() }
    } finally { $writer.Dispose(); $buffer.Dispose() }
} finally { $sourceImage.Dispose() }

Write-Output 'Generated Agent Czesiek PNG, ICO, ICNS and window icon.'
