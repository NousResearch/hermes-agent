#ifndef Py_LIMITED_API
#ifndef Py_PYDEBUG_H
#define Py_PYDEBUG_H
#ifdef __cplusplus
extern "C" {
#endif

PyAPI_DATA(int) Py_DebugFlag;
PyAPI_DATA(int) Py_VerboseFlag;
PyAPI_DATA(int) Py_QuietFlag;
PyAPI_DATA(int) Py_InteractiveFlag;
PyAPI_DATA(int) Py_InspectFlag;
PyAPI_DATA(int) Py_OptimizeFlag;
PyAPI_DATA(int) Py_NoSiteFlag;
PyAPI_DATA(int) Py_BytesWarningFlag;
PyAPI_DATA(int) Py_FrozenFlag;
PyAPI_DATA(int) Py_IgnoreEnvironmentFlag;
PyAPI_DATA(int) Py_DontWriteBytecodeFlag;
PyAPI_DATA(int) Py_NoUserSiteDirectory;
PyAPI_DATA(int) Py_UnbufferedStdioFlag;
PyAPI_DATA(int) Py_HashRandomizationFlag;
PyAPI_DATA(int) Py_IsolatedFlag;

#ifdef MS_WINDOWS
PyAPI_DATA(int) Py_LegacyWindowsFSEncodingFlag;
PyAPI_DATA(int) Py_LegacyWindowsStdioFlag;
#endif

/* this is a wrapper around getenv() that pays attention to
   Py_IgnoreEnvironmentFlag.  It should be used for getting variables like
   PYTHONPATH and PYTHONHOME from the environment */
PyAPI_DATA(char*) Py_GETENV(const char *name);

#ifdef __cplusplus
}
#endif
#endif /* !Py_PYDEBUG_H */
#endif /* Py_LIMITED_API */
