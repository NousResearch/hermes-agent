import { describe, expect, it } from 'vitest'

import { fileUrlToNativePath } from './file-url'

describe('fileUrlToNativePath', () => {
  it.each([
    ['file:///srv/report.html', '/srv/report.html'],
    ['file:///srv/report%20%231%3F.html', '/srv/report #1?.html'],
    ['file:///srv/100%25.txt', '/srv/100%.txt'],
    ['file:///srv/r%C3%A9sum%C3%A9.md', '/srv/résumé.md'],
    ['file:///srv/clip.mp4?download=1#t=30', '/srv/clip.mp4'],
    ['file://localhost/srv/report.html', '/srv/report.html'],
    ['FILE:///srv/report.html', '/srv/report.html'],
    // No host: a POSIX double-slash path, not UNC.
    ['file:////srv/share/report.html', '//srv/share/report.html'],
    // `\` separates only in drive and UNC paths; a POSIX name may contain one.
    ['file:///srv/name%5Cdraft.html', '/srv/name\\draft.html']
  ])('reads POSIX URL %s as %s', (url, path) => {
    expect(fileUrlToNativePath(url)).toBe(path)
  })

  it.each([
    ['file:///C:/work%20tree/report.html', 'C:\\work tree\\report.html'],
    ['file:///c:/Users/me/100%25.txt', 'c:\\Users\\me\\100%.txt'],
    ['file://localhost/C:/work/report.html', 'C:\\work\\report.html'],
    ['file:///C|/work/report.html', 'C:\\work\\report.html'],
    // `pathToFileUrl` encodes the drive colon; the decoded path still names a drive.
    ['file:///C%3A/work/report.html', 'C:\\work\\report.html'],
    ['file:///D:/a/../b/report.html', 'D:\\b\\report.html']
  ])('reads drive URL %s as %s', (url, path) => {
    expect(fileUrlToNativePath(url)).toBe(path)
  })

  it.each([
    ['file://server/share/report%20%231.html', '\\\\server\\share\\report #1.html'],
    ['file://NAS.example/share/report.html', '\\\\nas.example\\share\\report.html'],
    ['file://10.0.0.5/share/report.html', '\\\\10.0.0.5\\share\\report.html']
  ])('reads UNC URL %s as %s', (url, path) => {
    expect(fileUrlToNativePath(url)).toBe(path)
  })

  it.each([
    'file:///srv/project/..%2f..%2fetc/passwd',
    'file:///srv/project/..%2F..%2Fetc/passwd',
    'file:///C:/work/..%2f..%2fWindows/win.ini',
    'file:///C:/work/..%5c..%5cWindows/win.ini',
    'file:///C%3A%5CWindows/win.ini',
    'file://server/share/..%5Csecret.txt'
  ])('refuses the encoded separator in %s', url => {
    expect(fileUrlToNativePath(url)).toBeNull()
  })

  it.each(['file:///tmp/100%.txt', 'file:///tmp/%E0%A4%A.txt', 'file://%invalid/clip.mp4', 'not a url', '', 'file'])(
    'refuses malformed input %j',
    value => {
      expect(fileUrlToNativePath(value)).toBeNull()
    }
  )

  it.each(['https://example.com/report.html', 'blob:file:///srv/report.html', 'C:\\work\\report.html'])(
    'refuses non-file URL %s',
    value => {
      expect(fileUrlToNativePath(value)).toBeNull()
    }
  )
})
