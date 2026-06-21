#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using UnityEditor;
using UnityEngine;

namespace Hermes.UnityVrchatBridge
{
    [InitializeOnLoad]
    public sealed class HermesUnityVrchatBridgeWindow : EditorWindow
    {
        private const int DefaultPort = 17751;
        private const int MaxRequestBytes = 1024 * 64;
        private const string TrustedProjectsKey = "Hermes.UnityVrchatBridge.TrustedProjects";
        private const string AutoStartPrefix = "Hermes.UnityVrchatBridge.AutoStart.";
        private static readonly object LogsLock = new object();
        private static readonly List<BridgeLogEntry> RecentLogs = new List<BridgeLogEntry>();
        private static readonly object MainThreadLock = new object();
        private static readonly Queue<Action> MainThreadQueue = new Queue<Action>();
        private static HttpListener listener;
        private static Thread listenerThread;
        private static string token;
        private static int port = DefaultPort;
        private static bool running;
        private static int mainThreadId;

        static HermesUnityVrchatBridgeWindow()
        {
            mainThreadId = Thread.CurrentThread.ManagedThreadId;
            Application.logMessageReceived -= CaptureLog;
            Application.logMessageReceived += CaptureLog;
            EditorApplication.update -= PumpMainThreadQueue;
            EditorApplication.update += PumpMainThreadQueue;
            EditorApplication.delayCall += () =>
            {
                if (IsCurrentProjectTrusted() && AutoStartForCurrentProject && !running)
                {
                    StartBridge();
                }
            };
        }

        [MenuItem("Hermes/Unity VRChat Bridge")]
        public static void Open()
        {
            GetWindow<HermesUnityVrchatBridgeWindow>("Hermes Bridge");
        }

        public static void SelfTestForBatch()
        {
            TrustCurrentProject();
            StartBridge();
            try
            {
                Exception error = null;
                using (var done = new ManualResetEvent(false))
                {
                    var client = new Thread(() =>
                    {
                        try
                        {
                            Thread.Sleep(1000);
                            RequireContains(SelfTestRequest("GET", "/health", null, true, null), "\"ok\":true");
                            RequireContains(SelfTestRequest("GET", "/snapshot", null, true, null), "\"unityVersion\"");
                            RequireContains(SelfTestRequest("GET", "/selection", null, true, null), "\"ok\":true");
                            RequireContains(SelfTestRequest("GET", "/editor/capabilities", null, true, null), "\"capabilities\"");
                            RequireContains(SelfTestRequest("GET", "/project/packages", null, true, null), "\"packages\"");
                            RequireContains(SelfTestRequest("GET", "/scene/hierarchy?limit=20", null, true, null), "\"objects\"");
                            RequireContains(SelfTestRequest("GET", "/logs/recent?limit=5", null, true, null), "\"logs\"");
                            RequireContains(SelfTestRequest("POST", "/assets/search", "{\"filter\":\"t:DefaultAsset\",\"folders\":[\"Assets\"],\"limit\":5}", true, null), "\"assets\"");
                            RequireContains(SelfTestRequest("POST", "/assets/info", "{\"paths\":[\"Assets\"],\"includeDependencies\":true}", true, null), "\"assets\"");
                            RequireContains(SelfTestRequest("POST", "/menu/execute", "{\"menuPath\":\"VRChat SDK/Show Control Panel\",\"dryRun\":true}", true, null), "\"willExecute\":false");
                            RequireContains(SelfTestRequest("POST", "/menu/execute", "{\"menuPath\":\"VRChat SDK/Show Control Panel\",\"dryRun\":false}", true, null), "\"blockedAction\"");
                            RequireContains(SelfTestRequest("POST", "/operation/plan", "{\"operation\":\"asset_create\",\"targetPath\":\"Assets/New.asset\",\"dryRun\":true}", true, null), "\"willExecute\":false");
                            RequireContains(SelfTestRequest("POST", "/operation/plan", "{\"operation\":\"asset_create\",\"targetPath\":\"Assets/New.asset\",\"dryRun\":false}", true, null), "\"blockedAction\"");
                            RequireContains(SelfTestRequest("POST", "/plan/apply", "{\"operation\":\"avatar_preflight\",\"dryRun\":true}", true, null), "\"willApply\":false");
                            RequireContains(SelfTestRequest("POST", "/plan/apply", "{\"operation\":\"avatar_preflight\",\"dryRun\":false}", true, null), "\"blockedAction\"");
                            RequireStatus("GET", "/health", null, false, null, 401);
                            RequireStatus("GET", "/health", null, true, "http://example.com:80", 403);
                        }
                        catch (Exception ex)
                        {
                            error = ex;
                        }
                        finally
                        {
                            done.Set();
                        }
                    }) { IsBackground = true, Name = "HermesUnityVrchatBridgeSelfTest" };
                    client.Start();

                    var deadline = DateTime.UtcNow.AddSeconds(30);
                    while (!done.WaitOne(10))
                    {
                        PumpMainThreadQueue();
                        if (DateTime.UtcNow > deadline)
                        {
                            throw new TimeoutException("Unity bridge self-test timed out.");
                        }
                    }
                    PumpMainThreadQueue();
                }
                if (error != null) throw error;
                Debug.Log("HERMES_UNITY_VRCHAT_BRIDGE_SELFTEST_OK");
            }
            finally
            {
                StopBridge();
            }
        }

        private void OnGUI()
        {
            EditorGUILayout.LabelField("Hermes Unity VRChat Bridge", EditorStyles.boldLabel);
            EditorGUILayout.LabelField("Status", running ? "Running" : "Stopped");
            EditorGUILayout.LabelField("Port", port.ToString());
            EditorGUILayout.LabelField("Project", ProjectPath());
            EditorGUILayout.LabelField("Project Hash", ProjectHash());
            EditorGUILayout.LabelField("Trusted", IsCurrentProjectTrusted() ? "Yes" : "No");

            if (!IsCurrentProjectTrusted() && GUILayout.Button("Trust This Project"))
            {
                TrustCurrentProject();
            }
            if (IsCurrentProjectTrusted() && GUILayout.Button("Untrust This Project"))
            {
                UntrustCurrentProject();
                StopBridge();
            }

            var autoStart = AutoStartForCurrentProject;
            var nextAutoStart = EditorGUILayout.Toggle("Auto Start for This Project", autoStart);
            if (nextAutoStart != autoStart)
            {
                AutoStartForCurrentProject = nextAutoStart;
            }

            using (new EditorGUI.DisabledScope(!IsCurrentProjectTrusted()))
            {
                if (!running && GUILayout.Button("Start Bridge"))
                {
                    StartBridge();
                }
            }
            if (!running && !IsCurrentProjectTrusted())
            {
                EditorGUILayout.HelpBox("Trust this project before starting the localhost bridge.", MessageType.Warning);
            }
            if (running && GUILayout.Button("Stop Bridge"))
            {
                StopBridge();
            }

            EditorGUILayout.HelpBox(
                "The MVP exposes read-only endpoints on 127.0.0.1. SDK upload, package import, live menu execution, and destructive mutation are unavailable.",
                MessageType.Info);
        }

        private static void StartBridge()
        {
            if (running) return;
            if (!IsCurrentProjectTrusted())
            {
                Debug.LogError("Hermes Unity VRChat Bridge refused to start because this project is not trusted.");
                return;
            }
            token = GenerateToken();
            listener = null;
            for (var candidate = DefaultPort; candidate <= 17799; candidate++)
            {
                var next = new HttpListener();
                next.Prefixes.Add("http://127.0.0.1:" + candidate + "/");
                try
                {
                    next.Start();
                    listener = next;
                    port = candidate;
                    break;
                }
                catch
                {
                    try { next.Close(); } catch { }
                }
            }
            if (listener == null)
            {
                Debug.LogError("Hermes Unity VRChat Bridge could not bind a loopback port.");
                return;
            }
            running = true;
            WriteSession();
            listenerThread = new Thread(ListenLoop) { IsBackground = true, Name = "HermesUnityVrchatBridge" };
            listenerThread.Start();
            Debug.Log("HERMES_UNITY_VRCHAT_BRIDGE_START port=" + port);
        }

        private static void StopBridge()
        {
            running = false;
            try { listener?.Stop(); } catch { }
            try { listener?.Close(); } catch { }
            listener = null;
        }

        private static void ListenLoop()
        {
            while (running && listener != null)
            {
                try
                {
                    Handle(listener.GetContext());
                }
                catch
                {
                    if (!running) return;
                }
            }
        }

        private static void Handle(HttpListenerContext context)
        {
            try
            {
                if (!IsLoopback(context) || !IsOriginAllowed(context.Request.Headers["Origin"]))
                {
                    WriteJson(context, 403, "{\"ok\":false,\"error\":\"loopback origin required\"}");
                    return;
                }

                if (context.Request.Headers["X-Hermes-Bridge-Token"] != token)
                {
                    WriteJson(context, 401, "{\"ok\":false,\"error\":\"token required\"}");
                    return;
                }

                var path = context.Request.Url.AbsolutePath;
                if (context.Request.HttpMethod == "GET" && path == "/health")
                {
                    WriteJson(context, 200, "{\"ok\":true,\"bridge\":\"unity-vrchat-bridge\",\"version\":\"0.1.0\"}");
                    return;
                }
                if (context.Request.HttpMethod == "GET" && path == "/snapshot")
                {
                    WriteJson(context, 200, RunOnMainThread(BuildSnapshotJson));
                    return;
                }
                if (context.Request.HttpMethod == "GET" && path == "/selection")
                {
                    WriteJson(context, 200, RunOnMainThread(BuildSelectionJson));
                    return;
                }
                if (context.Request.HttpMethod == "GET" && path == "/editor/capabilities")
                {
                    WriteJson(context, 200, BuildCapabilitiesJson());
                    return;
                }
                if (context.Request.HttpMethod == "GET" && path == "/project/packages")
                {
                    WriteJson(context, 200, RunOnMainThread(BuildPackagesJson));
                    return;
                }
                if (context.Request.HttpMethod == "GET" && path == "/scene/hierarchy")
                {
                    WriteJson(context, 200, RunOnMainThread(() => BuildHierarchyJson(ReadLimit(context, 200))));
                    return;
                }
                if (context.Request.HttpMethod == "GET" && path == "/logs/recent")
                {
                    WriteJson(context, 200, BuildLogsJson(ReadLimit(context, 100)));
                    return;
                }
                if (context.Request.HttpMethod == "POST" && path == "/assets/search")
                {
                    var body = ReadBody(context);
                    WriteJson(context, 200, RunOnMainThread(() => BuildAssetSearchJson(body)));
                    return;
                }
                if (context.Request.HttpMethod == "POST" && path == "/assets/info")
                {
                    var body = ReadBody(context);
                    WriteJson(context, 200, RunOnMainThread(() => BuildAssetInfoJson(body)));
                    return;
                }
                if (context.Request.HttpMethod == "POST" && path == "/menu/execute")
                {
                    WriteJson(context, 200, BuildMenuPlanJson(ReadBody(context)));
                    return;
                }
                if (context.Request.HttpMethod == "POST" && path == "/operation/plan")
                {
                    WriteJson(context, 200, BuildOperationPlanJson(ReadBody(context)));
                    return;
                }
                if (context.Request.HttpMethod == "POST" && path == "/plan/apply")
                {
                    WriteJson(context, 200, BuildPlanApplyJson(ReadBody(context)));
                    return;
                }
                WriteJson(context, 404, "{\"ok\":false,\"error\":\"not found\"}");
            }
            catch (Exception ex)
            {
                WriteJson(context, 500, "{\"ok\":false,\"error\":\"" + Escape(ex.GetType().Name + ": " + ex.Message) + "\"}");
            }
        }

        private static bool IsLoopback(HttpListenerContext context)
        {
            return context.Request.RemoteEndPoint != null &&
                   IPAddress.IsLoopback(context.Request.RemoteEndPoint.Address);
        }

        private static bool IsOriginAllowed(string origin)
        {
            if (string.IsNullOrEmpty(origin)) return true;
            return origin.StartsWith("http://127.0.0.1:", StringComparison.OrdinalIgnoreCase) ||
                   origin.StartsWith("http://localhost:", StringComparison.OrdinalIgnoreCase);
        }

        private static string BuildSnapshotJson()
        {
            return "{"
                + "\"ok\":true,"
                + "\"unityVersion\":\"" + Escape(Application.unityVersion) + "\","
                + "\"projectPath\":\"" + Escape(ProjectPath()) + "\","
                + "\"dataPath\":\"" + Escape(Application.dataPath) + "\","
                + "\"activeScene\":\"" + Escape(UnityEditor.SceneManagement.EditorSceneManager.GetActiveScene().path) + "\","
                + "\"buildTarget\":\"" + Escape(EditorUserBuildSettings.activeBuildTarget.ToString()) + "\","
                + "\"isCompiling\":" + Bool(EditorApplication.isCompiling) + ","
                + "\"isPlaying\":" + Bool(EditorApplication.isPlaying)
                + "}";
        }

        private static string BuildSelectionJson()
        {
            var obj = Selection.activeObject;
            var path = obj == null ? "" : AssetDatabase.GetAssetPath(obj);
            return "{"
                + "\"ok\":true,"
                + "\"name\":\"" + Escape(obj == null ? "" : obj.name) + "\","
                + "\"type\":\"" + Escape(obj == null ? "" : obj.GetType().FullName) + "\","
                + "\"assetPath\":\"" + Escape(path) + "\","
                + "\"instanceId\":" + (obj == null ? 0 : obj.GetInstanceID())
                + "}";
        }

        private static string BuildCapabilitiesJson()
        {
            return "{"
                + "\"ok\":true,"
                + "\"bridge\":\"unity-vrchat-bridge\","
                + "\"capabilities\":["
                + "\"health\",\"snapshot\",\"selection\",\"logs_recent\",\"asset_search\","
                + "\"asset_info\",\"packages\",\"scene_hierarchy\",\"menu_execute_dry_run\","
                + "\"operation_plan_dry_run\",\"plan_apply_dry_run\""
                + "],"
                + "\"blockedActions\":[\"sdk_upload\",\"package_import\",\"delete_asset\",\"overwrite_asset\",\"manifest_mutation\",\"live_menu_execute\",\"live_plan_apply\"],"
                + "\"requiresProjectTrust\":true,"
                + "\"requiresToken\":true,"
                + "\"loopbackOnly\":true"
                + "}";
        }

        private static string BuildPackagesJson()
        {
            var packages = UnityEditor.PackageManager.PackageInfo.GetAllRegisteredPackages();
            var sb = new StringBuilder();
            sb.Append("{\"ok\":true,\"packages\":[");
            for (var i = 0; i < packages.Length; i++)
            {
                if (i > 0) sb.Append(",");
                var pkg = packages[i];
                sb.Append("{\"name\":\"").Append(Escape(pkg.name)).Append("\",");
                sb.Append("\"displayName\":\"").Append(Escape(pkg.displayName)).Append("\",");
                sb.Append("\"version\":\"").Append(Escape(pkg.version)).Append("\",");
                sb.Append("\"source\":\"").Append(Escape(pkg.source.ToString())).Append("\",");
                sb.Append("\"resolvedPath\":\"").Append(Escape(pkg.resolvedPath)).Append("\"}");
            }
            sb.Append("]}");
            return sb.ToString();
        }

        private static string BuildHierarchyJson(int limit)
        {
            var scene = UnityEditor.SceneManagement.EditorSceneManager.GetActiveScene();
            var roots = scene.GetRootGameObjects();
            var sb = new StringBuilder();
            var count = 0;
            sb.Append("{\"ok\":true,\"scene\":\"").Append(Escape(scene.path)).Append("\",\"objects\":[");
            for (var i = 0; i < roots.Length && count < limit; i++)
            {
                AppendHierarchyObject(sb, roots[i], roots[i].name, ref count, limit);
            }
            sb.Append("],\"truncated\":").Append(Bool(count >= limit)).Append("}");
            return sb.ToString();
        }

        private static void AppendHierarchyObject(StringBuilder sb, GameObject obj, string path, ref int count, int limit)
        {
            if (count >= limit) return;
            if (count > 0) sb.Append(",");
            var components = obj.GetComponents<Component>();
            sb.Append("{\"name\":\"").Append(Escape(obj.name)).Append("\",");
            sb.Append("\"path\":\"").Append(Escape(path)).Append("\",");
            sb.Append("\"activeSelf\":").Append(Bool(obj.activeSelf)).Append(",");
            sb.Append("\"tag\":\"").Append(Escape(obj.tag)).Append("\",");
            sb.Append("\"layer\":").Append(obj.layer).Append(",");
            sb.Append("\"components\":[");
            for (var i = 0; i < components.Length; i++)
            {
                if (i > 0) sb.Append(",");
                sb.Append("\"").Append(Escape(components[i] == null ? "Missing Script" : components[i].GetType().FullName)).Append("\"");
            }
            sb.Append("]}");
            count++;
            for (var i = 0; i < obj.transform.childCount && count < limit; i++)
            {
                var child = obj.transform.GetChild(i).gameObject;
                AppendHierarchyObject(sb, child, path + "/" + child.name, ref count, limit);
            }
        }

        private static string BuildLogsJson(int limit)
        {
            BridgeLogEntry[] entries;
            lock (LogsLock)
            {
                var count = Math.Min(Math.Max(limit, 1), RecentLogs.Count);
                entries = RecentLogs.GetRange(RecentLogs.Count - count, count).ToArray();
            }
            var sb = new StringBuilder();
            sb.Append("{\"ok\":true,\"logs\":[");
            for (var i = 0; i < entries.Length; i++)
            {
                if (i > 0) sb.Append(",");
                sb.Append(entries[i].ToJson());
            }
            sb.Append("]}");
            return sb.ToString();
        }

        private static string BuildAssetSearchJson(string body)
        {
            var filter = ExtractString(body, "filter");
            if (string.IsNullOrWhiteSpace(filter))
            {
                return "{\"ok\":false,\"error\":\"filter required\"}";
            }
            var limit = Math.Min(Math.Max(ExtractInt(body, "limit", 100), 1), 500);
            var folders = ExtractStringArray(body, "folders");
            if (folders.Length == 0) folders = new[] { "Assets" };
            for (var i = 0; i < folders.Length; i++)
            {
                if (!folders[i].Replace("\\", "/").StartsWith("Assets", StringComparison.Ordinal))
                {
                    return "{\"ok\":false,\"error\":\"folder must be under Assets\"}";
                }
            }
            var guids = AssetDatabase.FindAssets(filter, folders);
            var sb = new StringBuilder();
            sb.Append("{\"ok\":true,\"assets\":[");
            var count = Math.Min(limit, guids.Length);
            for (var i = 0; i < count; i++)
            {
                if (i > 0) sb.Append(",");
                var path = AssetDatabase.GUIDToAssetPath(guids[i]);
                sb.Append("{\"guid\":\"").Append(Escape(guids[i])).Append("\",");
                sb.Append("\"path\":\"").Append(Escape(path)).Append("\",");
                sb.Append("\"type\":\"").Append(Escape(AssetDatabase.GetMainAssetTypeAtPath(path)?.FullName ?? "")).Append("\"}");
            }
            sb.Append("],\"truncated\":").Append(Bool(guids.Length > count)).Append("}");
            return sb.ToString();
        }

        private static string BuildAssetInfoJson(string body)
        {
            var paths = ExtractStringArray(body, "paths");
            var includeDependencies = ExtractBool(body, "includeDependencies", false);
            var sb = new StringBuilder();
            sb.Append("{\"ok\":true,\"assets\":[");
            for (var i = 0; i < paths.Length && i < 100; i++)
            {
                if (i > 0) sb.Append(",");
                var path = paths[i].Replace("\\", "/");
                var main = AssetDatabase.LoadMainAssetAtPath(path);
                var type = AssetDatabase.GetMainAssetTypeAtPath(path);
                sb.Append("{\"path\":\"").Append(Escape(path)).Append("\",");
                sb.Append("\"exists\":").Append(Bool(main != null || AssetDatabase.IsValidFolder(path))).Append(",");
                sb.Append("\"guid\":\"").Append(Escape(AssetDatabase.AssetPathToGUID(path))).Append("\",");
                sb.Append("\"type\":\"").Append(Escape(type == null ? "" : type.FullName)).Append("\",");
                sb.Append("\"labels\":[");
                var labels = main == null ? new string[0] : AssetDatabase.GetLabels(main);
                for (var j = 0; j < labels.Length; j++)
                {
                    if (j > 0) sb.Append(",");
                    sb.Append("\"").Append(Escape(labels[j])).Append("\"");
                }
                sb.Append("],\"dependencies\":[");
                var deps = includeDependencies ? AssetDatabase.GetDependencies(path, false) : new string[0];
                for (var j = 0; j < deps.Length && j < 100; j++)
                {
                    if (j > 0) sb.Append(",");
                    sb.Append("\"").Append(Escape(deps[j])).Append("\"");
                }
                sb.Append("]}");
            }
            sb.Append("]}");
            return sb.ToString();
        }

        private static string BuildMenuPlanJson(string body)
        {
            var menuPath = ExtractString(body, "menuPath");
            var dryRun = ExtractBool(body, "dryRun", true);
            if (!dryRun)
            {
                return "{\"ok\":false,\"error\":\"menu execution is dry-run only in the MVP\",\"blockedAction\":\"menu_execute_live\"}";
            }
            return "{"
                + "\"ok\":true,"
                + "\"dryRun\":true,"
                + "\"menuPath\":\"" + Escape(menuPath) + "\","
                + "\"willExecute\":false,"
                + "\"blockedActions\":[\"menu_execute_live\",\"sdk_upload\",\"package_import\",\"destructive_project_mutation\"]"
                + "}";
        }

        private static string BuildPlanApplyJson(string body)
        {
            var operation = ExtractString(body, "operation");
            var dryRun = ExtractBool(body, "dryRun", true);
            if (!dryRun)
            {
                return "{\"ok\":false,\"error\":\"plan apply is dry-run only in the MVP\",\"blockedAction\":\"plan_apply_live\"}";
            }
            return "{"
                + "\"ok\":true,"
                + "\"dryRun\":true,"
                + "\"operation\":\"" + Escape(operation) + "\","
                + "\"willApply\":false,"
                + "\"requiresBackup\":true,"
                + "\"blockedActions\":[\"plan_apply_live\",\"sdk_upload\",\"package_import\",\"destructive_project_mutation\"]"
                + "}";
        }

        private static string BuildOperationPlanJson(string body)
        {
            var operation = ExtractString(body, "operation");
            var targetPath = ExtractString(body, "targetPath");
            var dryRun = ExtractBool(body, "dryRun", true);
            if (!dryRun)
            {
                return "{\"ok\":false,\"error\":\"operation execution is dry-run only in the MVP\",\"blockedAction\":\"operation_execute_live\"}";
            }
            if (!string.IsNullOrEmpty(targetPath) && !targetPath.Replace("\\", "/").StartsWith("Assets", StringComparison.Ordinal))
            {
                return "{\"ok\":false,\"error\":\"targetPath must be under Assets\",\"blockedAction\":\"path_outside_assets\"}";
            }
            return "{"
                + "\"ok\":true,"
                + "\"dryRun\":true,"
                + "\"operation\":\"" + Escape(operation) + "\","
                + "\"targetPath\":\"" + Escape(targetPath) + "\","
                + "\"willExecute\":false,"
                + "\"requiresBackup\":true,"
                + "\"blockedActions\":[\"operation_execute_live\",\"delete_asset\",\"overwrite_asset\",\"manifest_mutation\",\"sdk_upload\",\"package_import\"]"
                + "}";
        }

        private static int ReadLimit(HttpListenerContext context, int fallback)
        {
            var raw = context.Request.QueryString["limit"];
            int parsed;
            return int.TryParse(raw, out parsed) ? Math.Min(Math.Max(parsed, 1), 500) : fallback;
        }

        private static string ReadBody(HttpListenerContext context)
        {
            if (context.Request.ContentLength64 > MaxRequestBytes) return "";
            using (var reader = new StreamReader(context.Request.InputStream, context.Request.ContentEncoding ?? Encoding.UTF8))
            {
                var body = reader.ReadToEnd();
                return body.Length > MaxRequestBytes ? "" : body;
            }
        }

        private static void WriteJson(HttpListenerContext context, int status, string json)
        {
            var bytes = Encoding.UTF8.GetBytes(json);
            context.Response.StatusCode = status;
            context.Response.ContentType = "application/json";
            context.Response.ContentEncoding = Encoding.UTF8;
            context.Response.OutputStream.Write(bytes, 0, bytes.Length);
            context.Response.Close();
        }

        private static void PumpMainThreadQueue()
        {
            while (true)
            {
                Action action = null;
                lock (MainThreadLock)
                {
                    if (MainThreadQueue.Count == 0) return;
                    action = MainThreadQueue.Dequeue();
                }
                action();
            }
        }

        private static T RunOnMainThread<T>(Func<T> action)
        {
            if (Thread.CurrentThread.ManagedThreadId == mainThreadId)
            {
                return action();
            }

            T result = default(T);
            Exception error = null;
            using (var done = new ManualResetEvent(false))
            {
                lock (MainThreadLock)
                {
                    MainThreadQueue.Enqueue(() =>
                    {
                        try
                        {
                            result = action();
                        }
                        catch (Exception ex)
                        {
                            error = ex;
                        }
                        finally
                        {
                            done.Set();
                        }
                    });
                }

                if (!done.WaitOne(5000))
                {
                    throw new TimeoutException("Timed out waiting for Unity main thread.");
                }
            }

            if (error != null) throw error;
            return result;
        }

        private static string SelfTestRequest(string method, string path, string body, bool includeToken, string origin)
        {
            var request = (HttpWebRequest)WebRequest.Create("http://127.0.0.1:" + port + path);
            request.Method = method;
            request.Timeout = 5000;
            request.Proxy = null;
            request.KeepAlive = false;
            if (includeToken) request.Headers["X-Hermes-Bridge-Token"] = token;
            if (!string.IsNullOrEmpty(origin)) request.Headers["Origin"] = origin;
            if (body != null)
            {
                var bytes = Encoding.UTF8.GetBytes(body);
                request.ContentType = "application/json";
                request.ContentLength = bytes.Length;
                using (var stream = request.GetRequestStream())
                {
                    stream.Write(bytes, 0, bytes.Length);
                }
            }
            using (var response = (HttpWebResponse)request.GetResponse())
            using (var reader = new StreamReader(response.GetResponseStream()))
            {
                return reader.ReadToEnd();
            }
        }

        private static void RequireStatus(string method, string path, string body, bool includeToken, string origin, int status)
        {
            try
            {
                SelfTestRequest(method, path, body, includeToken, origin);
            }
            catch (WebException ex)
            {
                var response = ex.Response as HttpWebResponse;
                if (response != null && (int)response.StatusCode == status) return;
                throw;
            }
            throw new InvalidOperationException("Expected HTTP " + status + " for " + path);
        }

        private static void RequireContains(string value, string expected)
        {
            if (value == null || !value.Contains(expected))
            {
                throw new InvalidOperationException("Self-test response missing " + expected);
            }
        }

        private static void CaptureLog(string condition, string stackTrace, LogType type)
        {
            lock (LogsLock)
            {
                RecentLogs.Add(new BridgeLogEntry(condition, stackTrace, type.ToString(), DateTime.UtcNow.ToString("o")));
                if (RecentLogs.Count > 500) RecentLogs.RemoveRange(0, RecentLogs.Count - 500);
            }
        }

        private static string GenerateToken()
        {
            var bytes = new byte[32];
            RandomNumberGenerator.Fill(bytes);
            return Convert.ToBase64String(bytes);
        }

        private static void WriteSession()
        {
            var project = ProjectPath();
            var dir = Path.Combine(project, "Library", "HermesUnityBridge");
            Directory.CreateDirectory(dir);
            var json = "{"
                + "\"port\":" + port + ","
                + "\"token\":\"" + Escape(token) + "\","
                + "\"projectHash\":\"" + Escape(ProjectHash()) + "\""
                + "}";
            File.WriteAllText(Path.Combine(dir, "session.json"), json, Encoding.UTF8);
        }

        private static string ProjectPath()
        {
            return Directory.GetParent(Application.dataPath).FullName;
        }

        private static string ProjectHash()
        {
            return Sha256("sha256:", ProjectPath().Replace("\\", "/").ToLowerInvariant());
        }

        private static bool IsCurrentProjectTrusted()
        {
            var hash = ProjectHash();
            foreach (var entry in EditorPrefs.GetString(TrustedProjectsKey, "").Split('|'))
            {
                if (entry == hash) return true;
            }
            return false;
        }

        private static void TrustCurrentProject()
        {
            if (IsCurrentProjectTrusted()) return;
            var current = EditorPrefs.GetString(TrustedProjectsKey, "");
            var next = string.IsNullOrEmpty(current) ? ProjectHash() : current + "|" + ProjectHash();
            EditorPrefs.SetString(TrustedProjectsKey, next);
        }

        private static void UntrustCurrentProject()
        {
            var hash = ProjectHash();
            var kept = new List<string>();
            foreach (var entry in EditorPrefs.GetString(TrustedProjectsKey, "").Split('|'))
            {
                if (!string.IsNullOrEmpty(entry) && entry != hash) kept.Add(entry);
            }
            EditorPrefs.SetString(TrustedProjectsKey, string.Join("|", kept.ToArray()));
            AutoStartForCurrentProject = false;
        }

        private static bool AutoStartForCurrentProject
        {
            get { return EditorPrefs.GetBool(AutoStartPrefix + ProjectHash(), false); }
            set { EditorPrefs.SetBool(AutoStartPrefix + ProjectHash(), value); }
        }

        private static string Sha256(string prefix, string text)
        {
            using (var sha = SHA256.Create())
            {
                var bytes = sha.ComputeHash(Encoding.UTF8.GetBytes(text));
                var sb = new StringBuilder(prefix);
                foreach (var b in bytes) sb.Append(b.ToString("x2"));
                return sb.ToString();
            }
        }

        private static string ExtractString(string json, string key)
        {
            var match = Regex.Match(json ?? "", "\"" + Regex.Escape(key) + "\"\\s*:\\s*\"((?:\\\\.|[^\"])*)\"");
            return match.Success ? Unescape(match.Groups[1].Value) : "";
        }

        private static int ExtractInt(string json, string key, int fallback)
        {
            var match = Regex.Match(json ?? "", "\"" + Regex.Escape(key) + "\"\\s*:\\s*(\\d+)");
            int parsed;
            return match.Success && int.TryParse(match.Groups[1].Value, out parsed) ? parsed : fallback;
        }

        private static bool ExtractBool(string json, string key, bool fallback)
        {
            var match = Regex.Match(json ?? "", "\"" + Regex.Escape(key) + "\"\\s*:\\s*(true|false)", RegexOptions.IgnoreCase);
            return match.Success ? string.Equals(match.Groups[1].Value, "true", StringComparison.OrdinalIgnoreCase) : fallback;
        }

        private static string[] ExtractStringArray(string json, string key)
        {
            var match = Regex.Match(json ?? "", "\"" + Regex.Escape(key) + "\"\\s*:\\s*\\[(.*?)\\]", RegexOptions.Singleline);
            if (!match.Success) return new string[0];
            var values = new List<string>();
            foreach (Match item in Regex.Matches(match.Groups[1].Value, "\"((?:\\\\.|[^\"])*)\""))
            {
                values.Add(Unescape(item.Groups[1].Value));
            }
            return values.ToArray();
        }

        private static string Unescape(string value)
        {
            return (value ?? "").Replace("\\\"", "\"").Replace("\\\\", "\\");
        }

        private static string Escape(string value)
        {
            return (value ?? "").Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\r", "\\r").Replace("\n", "\\n");
        }

        private static string Bool(bool value)
        {
            return value ? "true" : "false";
        }

        private sealed class BridgeLogEntry
        {
            private readonly string condition;
            private readonly string stackTrace;
            private readonly string type;
            private readonly string timeUtc;

            public BridgeLogEntry(string condition, string stackTrace, string type, string timeUtc)
            {
                this.condition = condition;
                this.stackTrace = stackTrace;
                this.type = type;
                this.timeUtc = timeUtc;
            }

            public string ToJson()
            {
                return "{"
                    + "\"timeUtc\":\"" + Escape(timeUtc) + "\","
                    + "\"type\":\"" + Escape(type) + "\","
                    + "\"message\":\"" + Escape(condition) + "\","
                    + "\"stackTrace\":\"" + Escape(stackTrace) + "\""
                    + "}";
            }
        }
    }
}
#endif
