#ifndef Py_INTERNAL_PATHCONFIG_H
#define Py_INTERNAL_PATHCONFIG_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

PyAPI_FUNC(void) _PyPathConfig_ClearGlobal(void);
extern PyStatus _PyPathConfig_ReadGlobal(PyConfig *config);
extern PyStatus _PyPathConfig_UpdateGlobal(const PyConfig *config);
extern const wchar_t * _PyPathConfig_GetGlobalModuleSearchPath(void);

extern int _PyPathConfig_ComputeSysPath0(
    const PyWideStringList *argv,
    PyObject **path0);


#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_PATHCONFIG_H */
