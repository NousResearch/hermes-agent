#!/usr/bin/env python3
"""Generate the synthetic mock fixtures for tests/fixtures/ (post-cutover path).

All data is synthetic by construction — fake titles, round numbers,
mock-* identifiers, Jellyfin payloads in the schema's native PascalCase
as delivered by the real server's default application/json serializer
(live finding 17/10: camelCase only via Accept-profile negotiation,
which talaria does not send — see jellyfin__counts). No scraped live
data, no real usernames, no PII.

Run:  python3 tests/gen_fixtures.py     (from the py/ dir or anywhere)
Writes one .json per fixture into tests/fixtures/, overwriting, then
cross-checks every fixture the mock server routes reference so a missing
or stale fixture fails loudly instead of at request time.
"""

import json
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
FIXTURES = TESTS / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)


def W(name, data):
    (FIXTURES / (name + ".json")).write_text(
        json.dumps(data, indent=1, sort_keys=True) + "\n")


# ---------------------------------------------------------------- Sonarr v3
W("sonarr__initialize", {"apiKey": "mock-sonarr-key-from-initialize"})
W("sonarr__health", [])
W("sonarr__system_status",
  {"appName": "Sonarr", "instanceName": "Sonarr-mock", "version": "4.0.0-mock"})
W("sonarr__queue", {
    "page": 1, "pageSize": 10, "totalRecords": 2,
    "records": [
        {"id": 1, "status": "downloading", "title": "Mock Series Alpha S01E01",
         "size": 1000000000, "sizeleft": 400000000, "timeleft": "00:04:00",
         "estimatedCompletionTime": "2026-01-01T00:00:00Z",
         "statusMessages": []},
        {"id": 2, "status": "warning",
         "title": "Mock Series Alpha S01E02",
         "size": 900000000, "sizeleft": 500000000, "timeleft": "00:05:00",
         "statusMessages": [
             {"title": "warning", "messages": ["stalled download"]}]},
        {"id": 3, "status": "completed", "title": "Mock Series Alpha S01E03",
         "size": 800000000, "sizeleft": 0, "statusMessages": []}]})
W("sonarr__history", {
    "page": 1, "records": [
        {"id": 1, "eventType": 1, "sourceTitle": "Mock.Series.Alpha.S01E01",
         "date": "2026-01-01T00:00:00Z",
         "series": {"title": "Mock Series Alpha"},
         "episode": {"title": "Mock Episode One"},
         "quality": {"quality": {"name": "WEBDL-1080p"}}}]})
W("sonarr__rootfolder", [
    {"id": 1, "path": "/tv-mock", "accessible": True,
     "freeSpace": 500000000000, "totalSpace": 0}])
W("sonarr__qualityprofile", [{"id": 1, "name": "Mock Any"}])
W("sonarr__series", [
    {"id": 1, "title": "Mock Series Alpha", "monitored": True,
     "year": 2020,
     "statistics": {"seasonCount": 2, "episodeFileCount": 10,
                    "episodeCount": 12, "totalEpisodeCount": 12,
                    "sizeOnDisk": 50000000000}},
    {"id": 2, "title": "Mock Series Beta", "monitored": True,
     "year": 2019,
     "statistics": {"seasonCount": 1, "episodeFileCount": 6,
                    "episodeCount": 6, "totalEpisodeCount": 6,
                    "sizeOnDisk": 20000000000}},
    {"id": 3, "title": "Mock Series Gamma", "monitored": False,
     "year": 2018,
     "statistics": {"seasonCount": 1, "episodeFileCount": 0,
                    "episodeCount": 8, "totalEpisodeCount": 8,
                    "sizeOnDisk": 0}}])
W("sonarr__series_item",
  {"id": 1, "title": "Mock Series Alpha", "monitored": True,
   "tvdbId": 123456, "qualityProfileId": 1, "seasonFolder": True})
W("sonarr__deleted", {})
W("sonarr__series_lookup", [
    {"title": "Mock Series Alpha", "tvdbId": 123456, "year": 2020,
     "seasons": [{"seasonNumber": 1}, {"seasonNumber": 2}],
     "overview": "Synthetic fixture series."}])
W("sonarr__wanted_missing", {
    "page": 1, "pageSize": 50, "totalRecords": 2,
    "records": [
        {"series": {"title": "Mock Series Alpha"}, "seasonNumber": 1,
         "episodeNumber": 2, "title": "Mock Episode Two",
         "airDateUtc": "2026-01-05T00:00:00Z", "monitored": True,
         "hasFile": False},
        {"series": {"title": "Mock Series Beta"}, "seasonNumber": 1,
         "episodeNumber": 4, "title": "Mock Episode Four",
         "airDateUtc": "2026-01-07T00:00:00Z", "monitored": True,
         "hasFile": False}]})
W("sonarr__calendar", [
    {"series": {"title": "Mock Series Alpha"}, "seasonNumber": 1,
     "episodeNumber": 5, "title": "Mock Episode Five",
     "airDateUtc": "2026-01-09T00:00:00Z"},
    {"series": {"title": "Mock Series Alpha"}, "seasonNumber": 1,
     "episodeNumber": 6, "title": "Mock Episode Six",
     "airDateUtc": "2026-01-12T00:00:00Z"}])
W("sonarr__episodes", [
    {"id": 1, "episodeNumber": 1, "seasonNumber": 1, "monitored": True,
     "hasFile": True},
    {"id": 2, "episodeNumber": 2, "seasonNumber": 1, "monitored": True,
     "hasFile": False},
    {"id": 3, "episodeNumber": 3, "seasonNumber": 1, "monitored": True,
     "hasFile": False}])
W("sonarr__command", {"id": 1})

# ---------------------------------------------------------------- Radarr v3
W("radarr__initialize", {"apiKey": "mock-radarr-key-from-initialize"})
W("radarr__health", [])
W("radarr__system_status",
  {"appName": "Radarr", "instanceName": "Radarr-mock", "version": "5.0.0-mock"})
W("radarr__queue", {
    "page": 1, "pageSize": 10, "totalRecords": 1,
    "records": [
        {"id": 1, "status": "downloading", "title": "Mock Movie Alpha",
         "size": 8000000000, "sizeleft": 3000000000, "timeleft": "00:10:00",
         "statusMessages": []}]})
W("radarr__history", {
    "page": 1, "records": [
        {"id": 1, "eventType": 3, "sourceTitle": "Mock.Movie.Alpha.2021",
         "date": "2026-01-02T00:00:00Z",
         "movie": {"title": "Mock Movie Alpha"},
         "quality": {"quality": {"name": "BLURAY-1080p"}}}]})
W("radarr__rootfolder", [
    {"id": 1, "path": "/movies-mock", "accessible": True,
     "freeSpace": 700000000000, "totalSpace": 2000000000000}])
W("radarr__qualityprofile", [{"id": 1, "name": "Mock Any"}])
W("radarr__movie_list", [
    {"id": 1, "title": "Mock Movie Alpha", "monitored": True,
     "hasFile": True, "size": 8000000000, "sizeOnDisk": 8000000000,
     "qualityProfileId": 1, "isAvailable": True, "year": 2021,
     "added": "2026-01-01T00:00:00Z", "genres": ["Science Fiction"],
     "studio": "Mock Studio One"},
    {"id": 2, "title": "Mock Movie Beta", "monitored": True,
     "hasFile": False, "size": 0, "sizeOnDisk": 0,
     "qualityProfileId": 1, "isAvailable": False, "year": 2022,
     "added": "2026-01-03T00:00:00Z", "genres": ["Drama", "Thriller"],
     "studio": "Mock Studio Two"},
    {"id": 3, "title": "Mock Movie Gamma", "monitored": False,
     "hasFile": False, "size": 0, "sizeOnDisk": 0,
     "qualityProfileId": 1, "isAvailable": False, "year": 2020,
     "added": "2026-01-04T00:00:00Z", "genres": ["Drama"],
     "studio": "Mock Studio One"}])
W("radarr__movie_lookup", [
    {"title": "Mock Movie Alpha", "tmdbId": 100001, "year": 2021,
     "overview": "Synthetic fixture movie.",
     "images": [{"coverType": "poster",
                 "url": "http://127.0.0.1/mock.jpg"}]}])
W("radarr__movie_item",
  {"id": 1, "title": "Mock Movie Alpha", "tmdbId": 100001, "year": 2021,
   "monitored": True, "hasFile": False, "sizeOnDisk": 0,
   "qualityProfileId": 1, "rootFolderPath": "/movies-mock",
   "isAvailable": True})
W("radarr__deleted", {})
W("radarr__wanted_missing", {
    "page": 1, "pageSize": 50, "totalRecords": 1,
    "records": [
        {"title": "Mock Movie Beta", "year": 2022, "monitored": True,
         "hasFile": False}]})
W("radarr__calendar", [
    {"title": "Mock Movie Delta", "year": 2026,
     "digitalRelease": "2026-01-20T00:00:00Z"},
    {"title": "Mock Movie Epsilon", "year": 2026,
     "physicalRelease": "2026-01-25T00:00:00Z"}])
W("radarr__command", {"id": 2})

# ------------------------------------------------------------- Lidarr API v1
W("lidarr__initialize", {"apiKey": "mock-lidarr-key-from-initialize"})
W("lidarr__system_status", {"appName": "Lidarr", "version": "1.0.0-mock"})
W("lidarr__health", [])
W("lidarr__rootfolder", [
    {"id": 1, "path": "/music-mock", "accessible": True,
     "freeSpace": 300000000000, "totalSpace": 500000000000}])
W("lidarr__artists", [
    {"id": 1, "artistName": "Mock Artist", "monitored": True,
     "statistics": {"albumCount": 3, "trackCount": 30,
                    "trackFileCount": 25, "sizeOnDisk": 10000000000}}])
W("lidarr__search", [
    {"artist": "Mock Artist", "artistName": "Mock Artist",
     "title": "Mock Album", "albumTitle": "Mock Album",
     "type": "album", "albumType": "album",
     "releaseDate": "2020-05-01T00:00:00Z"}])
W("lidarr__queue", {
    "page": 1, "totalRecords": 1,
    "records": [
        {"id": 1, "status": "downloading",
         "title": "Mock Artist - Mock Album",
         "size": 200000000, "sizeleft": 100000000,
         "timeleft": "00:02:00", "statusMessages": []}]})

# ----------------------------------------------------------- Prowlarr API v1
W("prowlarr__initialize", {"apiKey": "mock-prowlarr-key-from-initialize"})
W("prowlarr__health", [])
W("prowlarr__system_status",
  {"appName": "Prowlarr", "version": "1.0.0-mock", "branch": "master"})
W("prowlarr__indexers", [
    {"id": 1, "name": "Mock Indexer One", "protocol": "torrent",
     "enable": True, "priority": 25, "fields": []},
    {"id": 2, "name": "Mock Indexer Two", "protocol": "usenet",
     "enable": True, "priority": 50, "fields": []},
    {"id": 3, "name": "Mock Indexer Three", "protocol": "torrent",
     "enable": False, "priority": 25, "fields": []}])
W("prowlarr__indexer_stats",
  {"indexers": [{"indexerId": 1, "indexerName": "Mock Indexer One",
                 "averageResponseTime": 100, "queries": 10, "grabs": 2,
                 "failedQueries": 0}]})
W("prowlarr__indexer_test",
  {"id": 1, "name": "Mock Indexer One", "isAvailable": True})
W("prowlarr__indexer_action",
  {"id": 1, "name": "Mock Indexer One", "enable": True, "protocol": "torrent",
   "priority": 25, "fields": [], "implementation": "Cardigann",
   "implementationName": "Cardigann", "configContract": "CardigannSettings"})
W("prowlarr__search", [
    {"indexer": "Mock Indexer One", "title": "Mock Result Alpha",
     "size": 1500000000, "protocol": "torrent"},
    {"indexer": "Mock Indexer Two", "title": "Mock Result Beta",
     "size": 900000000, "protocol": "usenet"}])
W("prowlarr__applications", [
    {"id": 1, "name": "Sonarr", "implementation": "Sonarr",
     "syncLevel": "fullSync"}])
W("prowlarr__app_created",
  {"id": 7, "name": "Sonarr", "implementation": "Sonarr",
   "syncLevel": "fullSync", "configContract": "SonarrSettings"})
W("prowlarr__app_action",
  {"id": 1, "name": "Sonarr", "implementation": "Sonarr"})
W("prowlarr__command", {"id": 3})
W("prowlarr__log", {
    "page": 1, "pageSize": 20, "totalRecords": 2,
    "records": [
        {"id": 1, "time": "2026-01-02T10:00:00Z", "level": "info",
         "message": "Mock log line one"},
        {"id": 2, "time": "2026-01-02T09:00:00Z", "level": "warn",
         "message": "Mock log line two"}]})

# ---------------------------------------------------------------- Bazarr
W("bazarr__health", [])
W("bazarr__system_status", {"data": {"version": "1.2.0-mock"}})
W("bazarr__settings", {"settings": {"general": {
    "enabled_languages": ["en", "de"],
    "serie_default_enabled": True, "movie_default_enabled": True,
    "single_language": False}}})
W("bazarr__history", {"data": [
    {"action": "synthetic-subtitle-download", "language": "en",
     "provider": "mockprovider",
     "seriesTitle": "Mock Series Alpha",
     "title": "Mock Series Alpha S01E01",
     "timestamp": "2026-01-01T10:30:00.000000Z"}]})
W("bazarr__search_result", [])
W("bazarr__episodes_wanted", {
    "data": [{"seriesTitle": "Mock Series Alpha",
              "episodeTitle": "S01E01",
              "missing_subtitles": [{"name": "English", "code2": "en",
                                   "code3": "eng", "forced": False, "hi": False}],
              "missingCount": 2, "missing": []}], "total": 2})
W("bazarr__movies_wanted", {
    "data": [{"movieTitle": "Mock Movie Beta",
              "title": "Mock Movie Beta",
              "missing_subtitles": [{"name": "German", "code2": "de",
                                   "code3": "deu", "forced": False, "hi": False}],
              "missingCount": 1}], "total": 1})

# ----------------------------------------------------------- Overseerr API v1
W("overseerr__status", {"version": "1.0.0-mock", "apiVersion": "v1",
                        "authMethods": ["local"]})
W("overseerr__request_count",
  {"pending": 1, "approved": 1, "declined": 0, "available": 0,
   "processing": 0, "deleted": 0})
W("overseerr__requests", {
    "page": 1, "pageSize": 20, "totalResults": 2, "totalPages": 1,
    "pageInfo": {"page": 1, "pageSize": 20, "results": 2, "pages": 1},
    "results": [
        {"id": 101, "type": "movie", "name": "Mock Movie Gamma",
         "status": 1,
         "requestedBy": {"id": 1, "displayName": "Mock User",
                         "email": "mock@users.test"},
         "media": {"id": 1, "mediaType": "movie", "tmdbId": 100003,
                   "status": 3, "title": "Mock Movie Gamma"}},
        {"id": 102, "type": "tv", "name": "Mock Series Delta",
         "status": 2,
         "requestedBy": {"id": 1, "displayName": "Mock User",
                         "email": "mock@users.test"},
         "media": {"id": 2, "mediaType": "tv", "tvdbId": 234567,
                   "status": 2, "title": "Mock Series Delta"}}]})
W("overseerr__request_item",
  {"id": 101, "type": "movie", "name": "Mock Movie Gamma", "status": 1,
   "createdAt": "2026-01-01T10:00:00.000Z",
   "requestedBy": {"id": 1, "displayName": "Mock User",
                   "email": "mock@users.test"},
   "media": {"id": 1, "mediaType": "movie", "tmdbId": 100003, "status": 3,
             "title": "Mock Movie Gamma"}})
W("overseerr__users", [{"id": 1, "displayName": "Mock User",
                        "requestCount": 2}])

# -------------------------------------------------------------- Jellyfin
# camelCase payloads as delivered by the server; .NET ticks are
# 1e7/second.
W("jellyfin__ping", "Jellyfin Server")
W("jellyfin__system_info",
  {"ServerName": "mock-jf", "Version": "10.10.7-mock", "Id": "mock-server-id",
   "OperatingSystem": "Linux"})
W("jellyfin__system_info_public",
  {"LocalAddress": "http://127.0.0.1:8096", "ServerName": "mock-jf",
   "Version": "10.10.7-mock", "ProductName": "Jellyfin Server",
   "OperatingSystem": "Linux", "Id": "mock-server-id",
   "StartupWizardCompleted": True})
W("jellyfin__users", [
    {"Id": "11aa22bb33cc44dd11aa22bb33cc44dd", "Name": "MockUserOne",
     "LastLoginDate": "2026-01-02T08:00:00.0000000Z", "HasPassword": True},
    {"Id": "22bb33cc44dd11aa22bb33cc44dd11aa", "Name": "MockUserTwo",
     "LastLoginDate": "2026-01-01T08:00:00.0000000Z", "HasPassword": True}])
W("jellyfin__sessions", [
    {"Id": "sess-active-1",
     "UserId": "11aa22bb33cc44dd11aa22bb33cc44dd",
     "UserName": "MockUserOne", "Client": "Jellyfin Web",
     "DeviceName": "Chrome", "DeviceType": "Desktop",
     "ApplicationVersion": "10.10.7", "IsActive": True,
     "LastActivityDate": "2026-01-02T09:00:00.0000000Z",
     "NowPlayingItem": {
         "Id": "88cd", "Name": "Mock Movie Alpha", "Type": "Movie",
         "ProductionYear": 2021, "SeriesName": None, "IndexNumber": None,
         "RunTimeTicks": 73200000000,
         "Genres": ["Science Fiction"]},
     "PlayState": {"PositionTicks": 18400000000, "IsPaused": False,
                   "IsMuted": False, "AudioStreamIndex": 1,
                   "SubtitleStreamIndex": 2, "PlayMethod": "Transcode",
                   "RepeatMode": "RepeatNone"},
     "TranscodingInfo": {"AudioCodec": "aac", "VideoCodec": "hevc",
                         "Container": "mp4", "IsVideoDirect": True,
                         "IsAudioDirect": False, "Bitrate": 8000000,
                         "CompletionPercentage": 24.6, "Width": 1920,
                         "Height": 1080,
                         "TranscodeReasons": ["ContainerNotSupported"]}},
    {"Id": "sess-idle-1",
     "UserId": "22bb33cc44dd11aa22bb33cc44dd11aa",
     "UserName": "MockUserTwo", "Client": "Android TV",
     "DeviceName": "Shield", "DeviceType": "Android",
     "ApplicationVersion": "0.16-mock", "IsActive": False,
     "LastActivityDate": "2026-01-01T20:00:00.0000000Z",
     "NowPlayingItem": None, "PlayState": None,
     "TranscodingInfo": None}])
W("jellyfin__virtual_folders", [
    # Mirrors a typical 5-library server: movies, tvshows,
    # music, boxsets, plus an unorganized tvshows folder.
    {"Name": "Mock Movies", "CollectionType": "movies",
     "Locations": ["/media-mock/movies"], "ItemId": "vf-movies",
     "LibraryOptions": {}, "RefreshProgress": 0.0,
     "RefreshStatus": ""},
    {"Name": "Mock Shows", "CollectionType": "tvshows",
     "Locations": ["/media-mock/tv"], "ItemId": "vf-shows",
     "LibraryOptions": {}, "RefreshProgress": 0.0,
     "RefreshStatus": ""},
    {"Name": "Mock Music", "CollectionType": "music",
     "Locations": ["/media-mock/music"], "ItemId": "vf-music",
     "LibraryOptions": {}, "RefreshProgress": 0.0,
     "RefreshStatus": ""},
    {"Name": "Mock Collections", "CollectionType": "boxsets",
     "Locations": [], "ItemId": "vf-collections",
     "LibraryOptions": {}, "RefreshProgress": 0.0,
     "RefreshStatus": ""},
    {"Name": "Mock Unorganized Shows", "CollectionType": "tvshows",
     "Locations": ["/media-mock/unorganized"], "ItemId": "vf-unorganized",
     "LibraryOptions": {}, "RefreshProgress": 0.0,
     "RefreshStatus": ""}])
W("jellyfin__counts", {
    # PascalCase keys = the 12.2.0 ItemCounts schema shape as delivered by
    # the default application/json serializer (live finding 17/10; spec
    # components.schemas.ItemCounts) — camelCase arrives only via Accept
    # profile negotiation, which talaria does not send.
    "MovieCount": 12, "SeriesCount": 3, "EpisodeCount": 40,
    "ArtistCount": 2, "AlbumCount": 8, "SongCount": 90,
    "MusicVideoCount": 0, "BoxSetCount": 1, "BookCount": 0,
    "TrailerCount": 0, "ProgramCount": 0, "ItemCount": 156})
W("jellyfin__latest", [
    {"Id": "77ee", "Name": "Mock Movie Alpha", "Type": "Movie",
     "ProductionYear": 2021, "DateCreated": "2026-01-03T00:00:00Z",
     "RunTimeTicks": 73200000000, "Genres": ["Science Fiction"]},
    {"Id": "77ef", "Name": "Mock Series Alpha S01E03", "Type": "Episode",
     "SeriesName": "Mock Series Alpha", "ProductionYear": 2020,
     "DateCreated": "2026-01-02T00:00:00Z", "RunTimeTicks": 18000000000,
     "Genres": ["Drama"]}])
W("jellyfin__resume", [
    {"Id": "88cd", "Name": "Mock Movie Alpha", "Type": "Movie",
     "ProductionYear": 2021, "RunTimeTicks": 73200000000,
     "UserData": {"PlaybackPositionTicks": 18400000000, "PlayCount": 3,
                  "Played": False}},
    {"Id": "91ff", "Name": "Mock Series Alpha S01E02", "Type": "Episode",
     "SeriesName": "Mock Series Alpha", "RunTimeTicks": 18000000000,
     "UserData": {"PlaybackPositionTicks": 6000000000, "PlayCount": 1,
                  "Played": False}}])
W("jellyfin__nextup", [
    {"Id": "92aa", "SeriesName": "Mock Series Alpha",
     "Name": "Mock Episode Four", "Type": "Episode", "IndexNumber": 4,
     "ParentIndexNumber": 1, "RunTimeTicks": 18000000000}])
W("jellyfin__activitylog", {
    "Items": [
        {"Id": 1, "Name": "MockUserOne has played Mock Movie Alpha",
         "ShortOverview": None, "Overview": None,
         "Type": "PlaybackStart", "ItemId": None,
         "Date": "2026-01-02T09:00:00.0000000Z",
         "UserId": "11aa22bb33cc44dd11aa22bb33cc44dd",
         "Severity": "Info"},
        {"Id": 2, "Name": "MockUserTwo has stopped Mock Series Alpha",
         "ShortOverview": None, "Overview": None,
         "Type": "PlaybackStop", "ItemId": None,
         "Date": "2026-01-01T21:00:00.0000000Z",
         "UserId": "22bb33cc44dd11aa22bb33cc44dd11aa",
         "Severity": "Info"}],
    "StartIndex": 0, "TotalRecordCount": 2})
W("jellyfin__items", {
    "Items": [
        {"Id": "88cd", "Name": "Mock Movie Alpha", "Type": "Movie",
         "ProductionYear": 2021, "RunTimeTicks": 73200000000,
         "Genres": ["Science Fiction"],
         "UserData": {"PlayCount": 3, "Played": True,
                      "LastPlayedDate": "2026-01-02T09:00:00.0000000Z"}},
        {"Id": "89ab", "Name": "Mock Movie Beta", "Type": "Movie",
         "ProductionYear": 2022, "RunTimeTicks": 66000000000,
         "Genres": ["Drama"],
         "UserData": {"PlayCount": 1, "Played": True,
                      "LastPlayedDate": "2026-01-05T22:00:00.0000000Z"}},
        {"Id": "91ff", "Name": "Mock Series Alpha S01E01", "Type": "Episode",
         "SeriesName": "Mock Series Alpha", "ProductionYear": 2020,
         "RunTimeTicks": 18000000000, "Genres": ["Drama"],
         "UserData": {"PlayCount": 2, "Played": True,
                      "LastPlayedDate": "2026-01-01T20:00:00.0000000Z"}},
        {"Id": "92cd", "Name": "Mock Series Alpha S01E02", "Type": "Episode",
         "SeriesName": "Mock Series Alpha", "ProductionYear": 2020,
         "RunTimeTicks": 18000000000, "Genres": ["Drama", "Thriller"],
         "UserData": {"PlayCount": 2, "Played": True,
                      "LastPlayedDate": "2026-01-01T21:00:00.0000000Z"}}],
    "TotalRecordCount": 4, "StartIndex": 0})
W("jellyfin__refresh", {})


# ------------------------------------------------------ route cross-check
def cross_check():
    sys.path.insert(0, str(TESTS))
    from mock_server import ROUTES      # noqa: E402
    referenced = sorted({fixture
                         for routes in ROUTES.values()
                         for _, fixture, _, _ in routes})
    written = sorted(p.stem for p in FIXTURES.glob("*.json"))
    missing = [f for f in referenced if f not in written]
    unused = [f for f in written if f not in referenced]
    if missing:
        raise SystemExit("FAIL: fixtures referenced by routes but not "
                         "generated: %s" % ", ".join(missing))
    return referenced, unused


if __name__ == "__main__":
    referenced, unused = cross_check()
    print("wrote %d fixtures covering %d routed fixtures; unreferenced: %s"
          % (len(list(FIXTURES.glob('*.json'))), len(referenced),
             ", ".join(unused) or "none"))
