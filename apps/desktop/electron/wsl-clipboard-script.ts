// STA is mandatory: System.Windows.Forms.Clipboard throws ThreadStateException
// off a single-threaded apartment. We emit base64 (not raw bytes) so the PNG
// survives stdout's text decoding intact, and write with [Console]::Out.Write
// to avoid a trailing newline.
const WSL_CLIPBOARD_IMAGE_SCRIPT = [
  'Add-Type -AssemblyName System.Windows.Forms,System.Drawing',
  '$img = [System.Windows.Forms.Clipboard]::GetImage()',
  'if ($null -eq $img) { exit 0 }',
  '$ms = New-Object System.IO.MemoryStream',
  '$img.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png)',
  '[Console]::Out.Write([System.Convert]::ToBase64String($ms.ToArray()))'
].join('\n')

export { WSL_CLIPBOARD_IMAGE_SCRIPT }
