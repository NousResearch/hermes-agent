$ids = @(25428,25832,26656,26924,31188,32808,32932,36620,37628,18576,5432,4124,5440)
foreach ($id in $ids) {
  taskkill /F /T /PID $id 2>$null | Out-Null
}
